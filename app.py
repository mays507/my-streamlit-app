# Y-Compass (와이컴퍼스) — Streamlit MVP+ (심사자 설득력 강화 버전)
# (1) CSV 업로드 기반 데이터 커버리지 확장
# (2) 결과 배지(데이터 기반/가이드 기반)
# (3) A/B/C 점수화 + 기여도 breakdown(설명가능성 강화)
# (4) 연도별 기준선 vs 내 성적 미니 차트(역축) + 점수/기여도 차트
#
# 실행:
#   streamlit run app.py
#
# 필요 패키지:
#   pip install streamlit pandas altair requests

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st

# =========================================================
# Page Config + Styling
# =========================================================
st.set_page_config(page_title="🧭 Y-Compass", page_icon="🧭", layout="wide")

st.markdown(
    """
<style>
/* Badge pills */
.badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.9rem;
  line-height: 1.2rem;
  border: 1px solid rgba(0,0,0,0.08);
}
.badge-data { background: rgba(16,185,129,0.14); color: rgb(6,95,70); }
.badge-guide { background: rgba(245,158,11,0.18); color: rgb(120,53,15); }
.badge-stable { background: rgba(59,130,246,0.14); color: rgb(30,64,175); }
.badge-fit { background: rgba(168,85,247,0.14); color: rgb(88,28,135); }
.badge-challenge { background: rgba(239,68,68,0.14); color: rgb(153,27,27); }

/* Cards */
.card-title { font-weight: 800; margin-bottom: 0.2rem; }
.small { color: rgba(0,0,0,0.6); font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Options
# =========================================================
ADMISSION_ROUTE = ["수시", "정시"]
SUSI_DETAIL = ["학생부교과", "학생부종합", "논술", "특기자(해당 시)"]
MAJOR_GROUPS = ["인문", "사회", "상경", "자연", "공학", "예체능", "융합/자유전공"]
ACTIVITY_PREF = ["사람(소통/리더십)", "데이터(분석/정량)", "글(에세이/스토리)", "현장(활동/프로젝트)"]
GOAL_PRIORITY = ["합격 안정성", "적성/흥미", "취업/진로 연계", "장학/비용", "지역/생활환경"]
CONSTRAINTS = ["지역(통학/거주)", "예산(비용)", "시간(병행 일정)", "가족/돌봄", "기타"]
CURRENT_STAGE = ["내신/수능 준비", "자기소개서/학생부 정리", "면접 준비", "논술 준비", "지원전략 최종 점검"]
EXTRACURRICULAR_LEVELS = ["낮음", "보통", "높음"]

CSV_REQUIRED_COLS = ["university", "major", "route", "year", "metric", "threshold"]
CSV_OPTIONAL_COLS = ["route_detail", "source", "note"]

# =========================================================
# Utilities
# =========================================================
def _nonempty(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) and s.strip() else ""


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def band_to_float(band: str) -> Optional[float]:
    """
    Convert UI band string to float-ish threshold.
    - "1.x" -> 1.5
    - "2.x" -> 2.5
    - "직접입력(예: 2.3)" -> parse as float
    - "모름/입력안함" -> None
    """
    band = _nonempty(band)
    if not band or "모름" in band:
        return None
    if band.endswith(".x"):
        try:
            return float(band.replace(".x", "")) + 0.5
        except Exception:
            return None
    try:
        return float(band)
    except Exception:
        return None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def coverage_badge_html(is_data_based: bool) -> str:
    if is_data_based:
        return '<span class="badge badge-data">데이터 기반 ✅</span>'
    return '<span class="badge badge-guide">가이드 기반 🟡</span>'


def band_badge_html(band: str) -> str:
    band = _nonempty(band)
    if band == "안정":
        return '<span class="badge badge-stable">안정</span>'
    if band == "적정":
        return '<span class="badge badge-fit">적정</span>'
    return '<span class="badge badge-challenge">도전</span>'


# =========================================================
# Data Handling: CSV -> normalized dataframe
# =========================================================
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in CSV_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필수 컬럼이 누락됨: {missing}")

    for c in CSV_OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = ""

    # normalize values
    df["university"] = df["university"].astype(str).str.strip()
    df["major"] = df["major"].astype(str).str.strip()
    df["route"] = df["route"].astype(str).str.strip()
    df["route_detail"] = df["route_detail"].astype(str).str.strip()
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()

    df["year"] = df["year"].apply(safe_int)
    df["threshold"] = df["threshold"].apply(safe_float)

    df["source"] = df["source"].astype(str).str.strip()
    df["note"] = df["note"].astype(str).str.strip()

    df = df.dropna(subset=["year", "threshold"])
    df = df[df["metric"].isin(["gpa", "mock"])]

    # Clean route variants
    df["route"] = df["route"].replace({"수시 ": "수시", "정시 ": "정시"})
    df = df[df["route"].isin(["수시", "정시"])]

    # Remove obvious invalid threshold
    df = df[(df["threshold"] > 0) & (df["threshold"] < 10)]

    return df


def _major_match(mj_input: str, mj_row: str) -> bool:
    mj_input = _nonempty(mj_input)
    mj_row = _nonempty(mj_row)
    if not mj_input or not mj_row:
        return True
    return (mj_input in mj_row) or (mj_row in mj_input)


def match_rows(
    df: pd.DataFrame,
    university: str,
    major_text: str,
    route: str,
    route_detail: str,
    metric: str,
    max_rows: int = 200,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    uni = _nonempty(university)
    mj = _nonempty(major_text)

    sub = df[df["university"] == uni]
    if mj:
        sub = sub[sub["major"].apply(lambda x: _major_match(mj, str(x)))]

    sub = sub[sub["route"] == route]
    sub = sub[sub["metric"] == metric]

    if route == "수시":
        rd = _nonempty(route_detail)
        if rd:
            exact = sub[sub["route_detail"] == rd]
            generic = sub[sub["route_detail"] == ""]
            sub = pd.concat([exact, generic], ignore_index=True).drop_duplicates()

    sub = sub.sort_values("year", ascending=True).head(max_rows)
    return sub


# =========================================================
# Explainable Scoring: Score -> Band, and A/B/C scoring
# =========================================================
@dataclass
class ScoreWeights:
    academics: float
    extracurricular: float
    constraints: float
    preference_fit: float


def extracurricular_score(level: str) -> float:
    return {"낮음": 20.0, "보통": 60.0, "높음": 90.0}.get(level, 60.0)


def constraints_penalty(constraints: List[str]) -> float:
    p = 0.0
    for c in constraints:
        if "시간" in c:
            p += 18
        elif "예산" in c:
            p += 15
        elif "지역" in c:
            p += 12
        else:
            p += 10
    return min(p, 45.0)


def preference_fit_score(activity_pref: List[str], route: str, route_detail: str) -> float:
    s = 50.0
    pref = " ".join(activity_pref)

    if route == "정시":
        if "데이터" in pref:
            s += 25
        if "글" in pref:
            s += 5
        if "사람" in pref:
            s += 5

    if route == "수시":
        if "학생부종합" in route_detail:
            if "글" in pref:
                s += 20
            if "현장" in pref:
                s += 15
            if "사람" in pref:
                s += 10
        elif "학생부교과" in route_detail:
            if "데이터" in pref:
                s += 20
        elif "논술" in route_detail:
            if "글" in pref:
                s += 20
            if "데이터" in pref:
                s += 10

    return clamp(s, 0.0, 100.0)


def academics_score(
    user_value: Optional[float],
    ref_series: Optional[pd.Series],
) -> Tuple[float, str, Optional[float]]:
    """
    Score 0..100. Lower grade is better.
    - If ref exists: anchor = most recent threshold (last year)
    - Return score, message, anchor
    """
    if user_value is None:
        return 50.0, "성적 입력이 없어 학업 점수는 중립(50)으로 처리했습니다.", None

    if ref_series is None or ref_series.empty:
        s = 100.0 - (user_value - 1.0) * 15.0
        s = clamp(s, 10.0, 90.0)
        return s, "데이터 커버리지 밖이라 절대값 기반(거친) 점수로 처리했습니다.", None

    anchor = float(ref_series.dropna().iloc[-1])
    diff = user_value - anchor

    if diff <= -0.2:
        s = 90.0
        msg = f"입력 성적({user_value:.1f})이 최근 기준선({anchor:.1f})보다 유리 → 학업 점수↑"
    elif diff <= 0.4:
        s = 65.0
        msg = f"입력 성적({user_value:.1f})이 최근 기준선({anchor:.1f}) 근처 → 학업 점수 중간"
    else:
        s = 35.0
        msg = f"입력 성적({user_value:.1f})이 최근 기준선({anchor:.1f})보다 불리 → 학업 점수↓"

    return s, msg, anchor


def normalize_weights(w: ScoreWeights) -> ScoreWeights:
    s = w.academics + w.extracurricular + w.constraints + w.preference_fit
    if s <= 0:
        return ScoreWeights(0.45, 0.25, 0.2, 0.1)
    return ScoreWeights(w.academics / s, w.extracurricular / s, w.constraints / s, w.preference_fit / s)


def total_score(
    w: ScoreWeights,
    acad: float,
    extra: float,
    penalty: float,
    fit: float,
) -> Tuple[float, Dict[str, float]]:
    wn = normalize_weights(w)

    contrib_acad = acad * wn.academics
    contrib_extra = extra * wn.extracurricular
    contrib_fit = fit * wn.preference_fit
    contrib_penalty = penalty * wn.constraints

    score = contrib_acad + contrib_extra + contrib_fit - contrib_penalty
    score = clamp(score, 0.0, 100.0)

    breakdown = {
        "학업(성적)": contrib_acad,
        "비교과": contrib_extra,
        "적합도(성향↔전형)": contrib_fit,
        "제약(감점)": -contrib_penalty,
        "총점": score,
    }
    return score, breakdown


def score_to_band(score: float) -> str:
    if score >= 75:
        return "안정"
    if score >= 55:
        return "적정"
    return "도전"


def abc_scores(
    base_score: float,
    constraints: List[str],
    priorities: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    A/B/C 점수화(설명가능 규칙):
    - base_score: 사용자 기본 적합도/가능성 총점
    - A(안정): 리스크 회피(제약 많을수록 A 권장 가산), +8
    - B(적정): 기준, +0
    - C(도전): 상향(제약 많을수록 감산), -10

    추가로 "목표 우선순위"가 '합격 안정성'이면 A 쪽 소폭 가산,
    '적성/흥미'가 있으면 B 쪽 소폭 가산,
    '취업/진로 연계'가 있으면 C/B 쪽 소폭 가산(도전 허용).
    """
    n_constraints = len(constraints)
    risk_factor = clamp(n_constraints / 4.0, 0.0, 1.0)  # 0~1

    p = " ".join(priorities or [])

    # small priority nudges
    a_nudge = 2.0 if "합격 안정성" in p else 0.0
    b_nudge = 1.5 if "적성/흥미" in p else 0.0
    c_nudge = 1.5 if "취업/진로 연계" in p else 0.0

    a = base_score + 8.0 + (risk_factor * 4.0) + a_nudge
    b = base_score + 0.0 + b_nudge
    c = base_score - 10.0 - (risk_factor * 5.0) + c_nudge

    out = {
        "A": {"label": "안정", "score": clamp(a, 0, 100)},
        "B": {"label": "적정", "score": clamp(b, 0, 100)},
        "C": {"label": "도전", "score": clamp(c, 0, 100)},
    }
    for k in out:
        out[k]["band"] = score_to_band(out[k]["score"])
    return out


# =========================================================
# OpenAI Responses API (optional)
# =========================================================
def openai_generate_plan(
    api_key: str,
    model: str,
    payload_json: Dict[str, Any],
    context_docs: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Uses Responses API. Requests JSON object output.
    """
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    prompt = f"""
너는 '대학 진학 AI 컨설턴트'다.

원칙(매우 중요):
- 사용자가 선택/입력한 전형을 우선 존중하되, 가능성/리스크/대안까지 함께 제시하라.
- 사실(전형요강/데이터)은 아래 [근거 문서]에 있는 내용만 사용하라.
- 근거 문서에 없는 수치/사실은 단정하지 말고 "일반 가이드"로 표현하라.
- 확률 단정 금지. 대신 안정/적정/도전 구간으로 표현하라.
- 8주 로드맵은 사용자가 선택한 전형과 현재 시점(월/주차)을 고려해
  "주차별 핵심 목표 1개 + 할 일 2~3개 + 산출물 1개"로 구조화하라.

출력은 반드시 아래 JSON 스키마로만 작성하라(다른 문장 금지).

JSON 스키마:
{{
  "summary_5lines": [string, string, string, string, string],
  "routes": {{
    "A": {{
      "title": "안정",
      "reasons": [string, string, string],
      "actions": [string, string, string, string, string],
      "risks": [string, string]
    }},
    "B": {{
      "title": "적정",
      "reasons": [string, string, string],
      "actions": [string, string, string, string, string],
      "risks": [string, string]
    }},
    "C": {{
      "title": "도전",
      "reasons": [string, string, string],
      "actions": [string, string, string, string, string],
      "risks": [string, string]
    }}
  }},
  "roadmap": [
    {{
      "week": number,
      "goal": string,
      "tasks": [string, string, string],
      "deliverable": string
    }}
  ],
  "evidence": [
    {{
      "title": string,
      "note": string
    }}
  ]
}}

[사용자 입력(JSON)]
{json.dumps(payload_json, ensure_ascii=False)}

[근거 문서]
{json.dumps(context_docs, ensure_ascii=False)}
""".strip()

    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        # Responses API supports JSON object response formats (structured outputs),
        # the model must be instructed to output JSON. (See response format docs)  # cite in write-up, not in code.
        "text": {"format": {"type": "json_object"}},
    }

    r = requests.post(url, headers=headers, json=body, timeout=75)
    r.raise_for_status()
    data = r.json()

    # Collect output text
    text_out = ""
    for out_item in data.get("output", []):
        for c in out_item.get("content", []):
            if c.get("type") in ("output_text", "text") and c.get("text"):
                text_out += c["text"]

    text_out = text_out.strip()
    if not text_out:
        raise ValueError("OpenAI 응답 텍스트가 비어있습니다.")

    # Parse JSON robustly
    try:
        return json.loads(text_out)
    except json.JSONDecodeError:
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text_out[start : end + 1])
        raise


# =========================================================
# Rule-based fallback
# =========================================================
def rule_based_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    route = payload.get("route", "수시")
    route_detail = payload.get("route_detail", "")
    major_group = payload.get("major_group", "")
    band = payload.get("band_label", "적정")

    summary = [
        f"희망 전형은 '{route}{(' - ' + route_detail) if route_detail else ''}'이며, 입력 조건 기반 구간은 '{band}'입니다.",
        f"관심 전공군은 '{major_group}'이고, 성향/제약을 전형 특성과 매칭했습니다.",
        "데이터 커버리지 내에서는 연도/출처 근거를 제시하고, 밖에서는 전략 가이드(정성)로 전환합니다.",
        "추천은 단정이 아니라 A/B/C 비교 구조로 제공합니다.",
        "8주 로드맵은 전형/현재 단계 기준으로 주차별 목표·할 일·산출물을 고정 출력합니다.",
    ]

    def mk_route(title: str) -> Dict[str, Any]:
        return {
            "title": title,
            "reasons": [
                "사용자 입력(성적/성향/제약)과 전형 특성의 정합성을 기준으로 구성했습니다.",
                "불확실 요소는 리스크로 분리하고 대안을 함께 제시합니다.",
                "실행 가능성을 높이기 위해 할 일을 액션 단위로 쪼갰습니다.",
            ],
            "actions": [
                "전형 요건 체크리스트 작성(필수/선택 분리)",
                "핵심 스토리 3개를 STAR로 정리(활동-역할-성과-배움)",
                "지원 조합 3개(A/B/C)로 분산 설계",
                "주 1회 피드백 루프(선생님/선배/AI)로 수정",
                "리스크 대비 대안 전형 1개 확보",
            ],
            "risks": [
                "데이터 범위 밖에서는 수치 예측을 제공하지 않습니다.",
                "전형별 산출물/일정이 촉박해질 수 있습니다.",
            ],
        }

    routes = {"A": mk_route("안정"), "B": mk_route("적정"), "C": mk_route("도전")}

    roadmap = []
    for w in range(1, 9):
        if route == "정시":
            goal = "실전 점수 안정화" if w <= 3 else ("약점 보완 집중" if w <= 6 else "실전 루틴 고정")
            tasks = [
                "기출/모의 1회분 풀이",
                "오답 원인 분류(개념/시간/실수)",
                "취약 단원 1개 보완",
            ]
            deliverable = f"Week {w}: 오답 분류표 + 취약 단원 계획"
        else:
            goal = "지원전략 확정" if w <= 2 else ("자소서/활동 정리" if w <= 5 else "면접/논술 대비")
            tasks = [
                "전형 요강 체크 + 제출물 목록화",
                "활동 3개 STAR 정리",
                "자소서/면접 질문 5개 초안 작성",
            ]
            deliverable = f"Week {w}: {route_detail or '수시'} 산출물 1종 초안"
        roadmap.append({"week": w, "goal": goal, "tasks": tasks, "deliverable": deliverable})

    evidence = [{"title": "일반 전략 가이드", "note": "키 미입력/데이터 범위 밖 시 룰베이스로 제공"}]
    return {"summary_5lines": summary, "routes": routes, "roadmap": roadmap, "evidence": evidence}


# =========================================================
# Session State
# =========================================================
if "df_data" not in st.session_state:
    st.session_state.df_data = pd.DataFrame()
if "result" not in st.session_state:
    st.session_state.result = None
if "payload" not in st.session_state:
    st.session_state.payload = None
if "evidence" not in st.session_state:
    st.session_state.evidence = []
if "score_breakdown" not in st.session_state:
    st.session_state.score_breakdown = None
if "abc" not in st.session_state:
    st.session_state.abc = None


# =========================================================
# Header
# =========================================================
st.title("🧭 Y-Compass (와이컴퍼스)")
st.caption("연세대 AX 캠프 Track 1 — 소그룹 챌린지 | 근거 기반 AI 진학 카운셀러 (MVP+)")

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("🔑 OpenAI (선택)")
    openai_key_default = st.secrets.get("OPENAI_API_KEY", "")
    openai_api_key = st.text_input("OpenAI API Key", value=openai_key_default, type="password", placeholder="sk-...")
    openai_model = st.text_input("모델", value="gpt-4.1-mini")

    st.divider()
    st.header("⚙️ 점수 가중치(설명가능성)")
    st.caption("총점은 0~100. 제약은 감점이며, 기여도(breakdown)를 공개합니다.")
    w_acad = st.slider("학업(성적)", 0.0, 1.0, 0.45, 0.05)
    w_extra = st.slider("비교과", 0.0, 1.0, 0.25, 0.05)
    w_const = st.slider("제약(감점)", 0.0, 1.0, 0.20, 0.05)
    w_fit = st.slider("전형-성향 적합도", 0.0, 1.0, 0.10, 0.05)
    weights = ScoreWeights(w_acad, w_extra, w_const, w_fit)

    st.divider()
    st.header("💳 상용화(티어) 초안")
    tier = st.selectbox("요금제(데모)", ["Free", "Basic", "Pro"])
    tier_desc = {
        "Free": "진단 + A/B/C 요약 + 2주 미니 체크리스트",
        "Basic": "A/B/C 상세 + 8주 로드맵 + 근거 보기",
        "Pro": "전형별 심화(자소서/면접 포인트) + PDF/저장/버전관리(컨셉)",
    }
    st.write(f"**{tier}**: {tier_desc[tier]}")

    st.divider()
    today = st.date_input("현재 시점(로드맵 기준)", value=date.today())

tabs = st.tabs(["🗃️ 데이터 업로드", "📝 진단 입력", "📌 결과", "📎 리포트/기획서"])

# =========================================================
# Tab 1: Data Upload
# =========================================================
with tabs[0]:
    st.subheader("🗃️ 입시 데이터 업로드 (CSV)")
    st.write("업로드된 데이터는 **근거(출처/연도) + 가능성 그래프 + 점수 산정(학업 비교)**에 반영됩니다.")

    template = (
        "university,major,route,route_detail,year,metric,threshold,source,note\n"
        "연세대,경영학과,수시,학생부종합,2022,gpa,2.0,입학처,예시\n"
        "연세대,경영학과,수시,학생부종합,2023,gpa,2.1,입학처,예시\n"
        "연세대,경영학과,수시,학생부종합,2024,gpa,2.2,입학처,예시\n"
        "연세대,경영학과,정시,,2024,mock,1.6,입학처,예시\n"
    )

    col_a, col_b = st.columns([1.2, 1.0], gap="large")
    with col_a:
        with st.expander("CSV 템플릿(권장 컬럼) 보기", expanded=True):
            st.code(template, language="text")

    with col_b:
        st.download_button(
            "⬇️ CSV 템플릿 다운로드",
            data=template.encode("utf-8"),
            file_name="y_compass_admissions_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("CSV 업로드", type=["csv"])
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            df = normalize_df(df_raw)
            st.session_state.df_data = df
            st.success(f"업로드 성공! rows={len(df):,}")
            st.dataframe(df.head(30), use_container_width=True)
        except Exception as e:
            st.error("CSV 파싱/정규화 실패")
            st.caption(str(e))

    if st.session_state.df_data is not None and not st.session_state.df_data.empty:
        st.divider()
        st.markdown("#### 데이터 커버리지 점검(빠른 필터)")
        df = st.session_state.df_data
        u = st.selectbox("대학", sorted(df["university"].unique().tolist()))
        majors = sorted(df[df["university"] == u]["major"].unique().tolist())
        m = st.selectbox("학과", majors)
        r = st.selectbox("전형(수시/정시)", ["수시", "정시"])
        rd = ""
        if r == "수시":
            rd_list = sorted(
                df[(df["university"] == u) & (df["major"] == m) & (df["route"] == "수시")]["route_detail"]
                .fillna("")
                .unique()
                .tolist()
            )
            rd = st.selectbox("수시 세부", rd_list)
        metric = st.selectbox("기준선 유형(metric)", ["gpa", "mock"])

        matched = match_rows(df, u, m, r, rd, metric)
        st.write(f"매칭 결과: {len(matched)} rows")
        st.dataframe(matched, use_container_width=True)

# =========================================================
# Tab 2: Intake
# =========================================================
with tabs[1]:
    st.subheader("📝 진단 입력 (3분)")
    st.caption("희망 전형 입력 → 성적/성향/제약 기반 점수화 → 데이터 기반이면 근거/그래프까지 생성")

    df_all = st.session_state.df_data if st.session_state.df_data is not None else pd.DataFrame()
    use_df = not df_all.empty

    with st.form("intake", clear_on_submit=False):
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("#### 1) 희망 전형 입력(피드백 반영 핵심)")
            grade_status = st.selectbox("학년/상태", ["고3", "N수(재수/삼수)", "고2(미리보기)"])
            route = st.selectbox("수시/정시", ADMISSION_ROUTE)
            route_detail = ""
            if route == "수시":
                route_detail = st.selectbox("수시 세부 전형", SUSI_DETAIL)

            desired_university = st.text_input("희망 대학(권장)", placeholder="예: 연세대")
            desired_major = st.text_input("희망 학과(권장)", placeholder="예: 경영학과")
            desired_text = st.text_input("희망 전형/학과/대학(자유 입력)", placeholder="예: 연세대 경영학과 학생부종합")

            st.markdown("#### 2) 성적 입력(구간)")
            gpa_band = st.selectbox("내신 등급", ["모름/입력안함", "1.x", "2.x", "3.x", "4.x", "5.x", "직접입력"])
            gpa_direct = st.text_input("내신 직접 입력(선택)", placeholder="예: 2.3") if gpa_band == "직접입력" else ""
            mock_band = st.selectbox("모의고사 등급/환산", ["모름/입력안함", "1.x", "2.x", "3.x", "4.x", "5.x", "직접입력"])
            mock_direct = st.text_input("모의 직접 입력(선택)", placeholder="예: 2.1") if mock_band == "직접입력" else ""

        with c2:
            st.markdown("#### 3) 성향/비교과/제약")
            major_group = st.selectbox("관심 전공군", MAJOR_GROUPS)
            activity_pref = st.multiselect("선호 활동/강점", ACTIVITY_PREF, default=[ACTIVITY_PREF[0]])
            extracurricular = st.select_slider("비교과 강도(자가평가)", options=EXTRACURRICULAR_LEVELS, value="보통")
            priorities = st.multiselect("목표 우선순위(최대 2)", GOAL_PRIORITY, default=[GOAL_PRIORITY[0], GOAL_PRIORITY[1]])
            constraints = st.multiselect("제약", CONSTRAINTS, default=[])

            current_stage = st.selectbox("현재 단계(로드맵 기준)", CURRENT_STAGE)
            notes = st.text_area("추가 메모(선택)", placeholder="예: 논술 병행 / 통학 제약 / 면접이 불안")
            st.info("개인정보(학교/실명/연락처 등) 입력 금지. 추천은 참고용입니다.")

        go = st.form_submit_button("결과 생성", type="primary")

    if go:
        gpa_val = band_to_float(gpa_direct if gpa_band == "직접입력" else gpa_band)
        mock_val = band_to_float(mock_direct if mock_band == "직접입력" else mock_band)

        metric = "mock" if route == "정시" else "gpa"
        user_metric_value = mock_val if metric == "mock" else gpa_val

        # Allow fallback parsing from free text
        uni = _nonempty(desired_university)
        mj = _nonempty(desired_major)
        if not uni and _nonempty(desired_text):
            uni = _nonempty(desired_text).split()[0]
        if not mj and _nonempty(desired_text):
            # heuristic: second token as major if exists
            toks = _nonempty(desired_text).split()
            if len(toks) >= 2:
                mj = toks[1]

        matched = pd.DataFrame()
        if use_df and uni and mj:
            matched = match_rows(df_all, uni, mj, route, route_detail, metric)

        is_data_based = (matched is not None) and (not matched.empty)

        ref_series = matched.sort_values("year")["threshold"] if is_data_based else None
        acad_s, acad_msg, anchor = academics_score(user_metric_value, ref_series)

        extra_s = extracurricular_score(extracurricular)
        penalty = constraints_penalty(constraints)
        fit_s = preference_fit_score(activity_pref, route, route_detail)

        tot, breakdown = total_score(weights, acad_s, extra_s, penalty, fit_s)
        band = score_to_band(tot)
        abc = abc_scores(tot, constraints, priorities[:2])

        payload = {
            "today": str(today),
            "grade_status": grade_status,
            "route": route,
            "route_detail": route_detail,
            "desired_university": uni,
            "desired_major": mj,
            "desired_text": desired_text,
            "major_group": major_group,
            "gpa_value": gpa_val,
            "mock_value": mock_val,
            "metric_used": metric,
            "metric_value": user_metric_value,
            "activity_pref": activity_pref,
            "extracurricular_level": extracurricular,
            "priorities": priorities[:2],
            "constraints": constraints,
            "current_stage": current_stage,
            "notes": notes,
            "band_label": band,
            "coverage_is_data": is_data_based,
            "score_total": float(tot),
            "score_breakdown": breakdown,
            "abc_scores": abc,
            "scoring_notes": {
                "academics": acad_msg,
                "fit": "전형-성향 적합도는 선택 성향과 전형 특성 매칭으로 산출했습니다.",
                "constraints": "제약은 감점으로 적용되며(가중치 반영), 많을수록 리스크가 커집니다.",
            },
            "data_anchor_threshold": anchor,
        }

        st.session_state.payload = payload
        st.session_state.score_breakdown = breakdown
        st.session_state.abc = abc

        # Evidence docs (RAG context)
        evidence_docs: List[Dict[str, str]] = []
        if is_data_based:
            tail = matched.sort_values("year").tail(12)
            for _, row in tail.iterrows():
                title = f"{row['university']} {row['major']} | {row['route']}{(' - ' + row['route_detail']) if row['route_detail'] else ''} | {int(row['year'])}"
                note = (
                    f"metric={row['metric']} threshold={row['threshold']} | "
                    f"source={row.get('source','')} | note={row.get('note','')}"
                )
                evidence_docs.append({"title": title, "note": note})
        else:
            evidence_docs.append(
                {
                    "title": "데이터 미보유(가이드 기반)",
                    "note": "해당 대학/학과/전형의 업로드 데이터가 없어 수치 기반 예측을 제공하지 않고, 전형 특성 기반 전략 가이드로 안내합니다.",
                }
            )

        st.session_state.evidence = evidence_docs

        # Generate plan
        with st.spinner("A/B/C 추천 + 8주 로드맵 생성 중..."):
            try:
                if _nonempty(openai_api_key):
                    plan = openai_generate_plan(
                        api_key=openai_api_key.strip(),
                        model=openai_model.strip(),
                        payload_json=payload,
                        context_docs=evidence_docs,
                    )
                else:
                    plan = rule_based_plan(payload)

                plan["_meta"] = {
                    "coverage_is_data": is_data_based,
                    "score_total": payload["score_total"],
                    "score_breakdown": payload["score_breakdown"],
                    "abc_scores": payload["abc_scores"],
                    "academics_msg": acad_msg,
                }

                st.session_state.result = plan
                st.success("완료! '📌 결과' 탭에서 확인해줘.")
            except Exception as e:
                st.session_state.result = None
                st.error("생성 실패(키/모델/네트워크/JSON 형식) 확인")
                st.caption(str(e))

# =========================================================
# Charts (Altair helpers)
# =========================================================
def chart_threshold_vs_user(chart_df: pd.DataFrame) -> alt.Chart:
    """
    chart_df: columns = ['year','threshold','user_value']
    Lower is better -> reverse Y axis.
    """
    base = alt.Chart(chart_df).encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("value:Q", title="등급(낮을수록 유리)", scale=alt.Scale(reverse=True)),
        color=alt.Color("series:N", title=""),
        tooltip=[
            alt.Tooltip("year:O", title="연도"),
            alt.Tooltip("series:N", title="항목"),
            alt.Tooltip("value:Q", title="값", format=".2f"),
        ],
    )

    long_df = chart_df.melt(id_vars=["year"], value_vars=["threshold", "user_value"], var_name="series", value_name="value")
    long_df["series"] = long_df["series"].replace({"threshold": "기준선(threshold)", "user_value": "내 성적(user)"})

    line = base.mark_line(point=True).transform_lookup(
        lookup="year",
        from_=alt.LookupData(long_df, "year", ["series", "value"]),
    )

    # transform_lookup + base encodes series/value, but lookup duplicates; simpler:
    line = alt.Chart(long_df).mark_line(point=True).encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("value:Q", title="등급(낮을수록 유리)", scale=alt.Scale(reverse=True)),
        color=alt.Color("series:N", title=""),
        tooltip=[
            alt.Tooltip("year:O", title="연도"),
            alt.Tooltip("series:N", title="항목"),
            alt.Tooltip("value:Q", title="값", format=".2f"),
        ],
    )

    return line.properties(height=260)


def chart_breakdown(breakdown: Dict[str, float]) -> alt.Chart:
    keys = ["학업(성적)", "비교과", "적합도(성향↔전형)", "제약(감점)"]
    rows = [{"요소": k, "기여도": float(breakdown.get(k, 0.0))} for k in keys]
    df = pd.DataFrame(rows)

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("기여도:Q", title="기여도(가중치 반영)"),
            y=alt.Y("요소:N", title=""),
            tooltip=[alt.Tooltip("요소:N"), alt.Tooltip("기여도:Q", format=".2f")],
        )
        .properties(height=220)
    )


def chart_abc_scores(abc: Dict[str, Dict[str, Any]]) -> alt.Chart:
    df = pd.DataFrame(
        [
            {"경로": "A(안정)", "점수": abc["A"]["score"], "구간": abc["A"]["band"]},
            {"경로": "B(적정)", "점수": abc["B"]["score"], "구간": abc["B"]["band"]},
            {"경로": "C(도전)", "점수": abc["C"]["score"], "구간": abc["C"]["band"]},
        ]
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("경로:N", title="경로(A/B/C)"),
            y=alt.Y("점수:Q", title="점수(0~100)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("경로:N"), alt.Tooltip("점수:Q", format=".1f"), alt.Tooltip("구간:N")],
        )
        .properties(height=260)
    )

# =========================================================
# Tab 3: Results
# =========================================================
with tabs[2]:
    st.subheader("📌 결과")

    if st.session_state.result is None or st.session_state.payload is None:
        st.info("먼저 '📝 진단 입력'에서 결과를 생성해줘.")
    else:
        payload = st.session_state.payload
        plan = st.session_state.result
        meta = plan.get("_meta", {})

        # --- Section 1
        st.markdown("## 섹션 1 — 내가 원하는 전형 가능성 카드")

        col1, col2, col3 = st.columns([1.2, 1.2, 2.4], gap="large")

        with col1:
            with st.container(border=True):
                st.markdown('<div class="card-title">커버리지</div>', unsafe_allow_html=True)
                st.markdown(coverage_badge_html(payload["coverage_is_data"]), unsafe_allow_html=True)
                st.markdown('<div class="small">데이터 기반이면 연도/출처 근거 및 그래프 제공</div>', unsafe_allow_html=True)

        with col2:
            with st.container(border=True):
                st.markdown('<div class="card-title">가능성 구간(안정/적정/도전)</div>', unsafe_allow_html=True)
                st.markdown(band_badge_html(payload["band_label"]), unsafe_allow_html=True)
                st.metric("총점(0~100)", f"{payload['score_total']:.1f}", help="가중치 기반 설명가능 점수")
                if payload.get("metric_value") is None:
                    st.caption("성적 미입력 → 학업 점수는 중립 처리")
                else:
                    st.caption(f"사용 지표: {payload['metric_used']} | 입력값: {payload['metric_value']:.2f}")

        with col3:
            with st.container(border=True):
                st.markdown('<div class="card-title">왜 이 구간인가? (설명가능 점수화)</div>', unsafe_allow_html=True)
                st.write(meta.get("academics_msg", ""))

                bd = payload["score_breakdown"]
                st.altair_chart(chart_breakdown(bd), use_container_width=True)

        st.divider()

        # --- Mini chart & table when data-based
        df_all = st.session_state.df_data if st.session_state.df_data is not None else pd.DataFrame()
        if payload["coverage_is_data"] and not df_all.empty:
            st.markdown("### 근거 시각화(연도별 기준선 vs 내 성적)")
            metric = payload["metric_used"]
            uni = payload.get("desired_university", "")
            mj = payload.get("desired_major", "")
            route = payload.get("route", "수시")
            rd = payload.get("route_detail", "")

            matched = match_rows(df_all, uni, mj, route, rd, metric)
            if matched is not None and not matched.empty and payload.get("metric_value") is not None:
                chart_df = matched.sort_values("year")[["year", "threshold"]].copy()
                chart_df["user_value"] = float(payload["metric_value"])

                st.dataframe(chart_df, use_container_width=True)

                st.altair_chart(chart_threshold_vs_user(chart_df), use_container_width=True)
                st.caption("※ 등급 기준: **낮을수록 유리**. (그래프는 참고용이며, 단정적 합격 예측이 아님)")
            elif matched is not None and not matched.empty:
                st.info("매칭 데이터는 있으나, 성적 입력이 없어 비교 그래프를 만들 수 없습니다.")
            else:
                st.info("시각화할 매칭 데이터가 없습니다(학과명/전형 키워드 확인).")
        else:
            st.markdown("### 근거 시각화")
            st.info("현재는 **가이드 기반**이거나(데이터 미보유), 데이터 업로드가 없어 그래프를 표시하지 않습니다.")

        st.divider()

        # --- A/B/C Score Chart (요구사항 3)
        st.markdown("## 섹션 2 — A/B/C 추천 점수화(설명가능성 강화)")
        abc = payload.get("abc_scores") or {}
        if abc:
            st.altair_chart(chart_abc_scores(abc), use_container_width=True)
            st.caption("A/B/C 점수는 **총점(기본 적합도)**를 기준으로, 제약/목표 우선순위를 반영해 *경로 난이도*를 조정해 산출합니다.")
        else:
            st.warning("A/B/C 점수화 결과가 없습니다.")

        st.divider()

        # --- A/B/C cards
        st.markdown("## 섹션 3 — AI 추천 전형/전략 TOP3 (A/B/C)")
        routes = plan.get("routes", {})
        cols = st.columns(3, gap="large")
        keys = ["A", "B", "C"]
        title_map = {"A": "A: 안정", "B": "B: 적정", "C": "C: 도전"}

        for i, k in enumerate(keys):
            r = routes.get(k, {})
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"### {title_map[k]}")
                    if abc and k in abc:
                        st.caption(f"점수: {abc[k]['score']:.1f} / 100 · 구간: {abc[k]['band']}")
                    else:
                        st.caption(r.get("title", ""))

                    st.markdown("**추천 이유(3)**")
                    for x in (r.get("reasons") or [])[:3]:
                        st.write(f"- {x}")

                    st.markdown("**준비 액션(5)**")
                    for x in (r.get("actions") or [])[:5]:
                        st.write(f"- {x}")

                    st.markdown("**리스크/가정(2)**")
                    for x in (r.get("risks") or [])[:2]:
                        st.write(f"- {x}")

        st.divider()

        # --- 8-week Roadmap
        st.markdown("## 섹션 4 — 8주 로드맵")
        roadmap = plan.get("roadmap", [])
        if not roadmap:
            st.warning("로드맵이 비어있습니다.")
        else:
            for item in roadmap[:8]:
                w = item.get("week")
                with st.expander(f"Week {w} — {item.get('goal','')}", expanded=(w == 1)):
                    st.markdown("**할 일(2~3)**")
                    for t in (item.get("tasks") or [])[:3]:
                        st.write(f"- {t}")
                    st.markdown("**산출물**")
                    st.write(item.get("deliverable", ""))

        st.divider()

        # --- Evidence
        st.markdown("### 근거 보기(출처)")
        evs = plan.get("evidence", []) or st.session_state.evidence or []
        if evs:
            for ev in evs[:15]:
                with st.expander(ev.get("title", "근거")):
                    st.write(ev.get("note", ""))
        else:
            st.caption("표시할 근거가 없습니다.")

        st.divider()

        # --- Download
        st.markdown("### 결과 저장(제출/시연용)")
        report = {
            "payload": payload,
            "summary_5lines": plan.get("summary_5lines", []),
            "routes": plan.get("routes", {}),
            "roadmap": plan.get("roadmap", []),
            "evidence": plan.get("evidence", []),
            "meta": plan.get("_meta", {}),
        }
        st.download_button(
            "📄 결과 리포트 다운로드 (.json)",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="y_compass_report.json",
            mime="application/json",
        )

# =========================================================
# Tab 4: Report / Spec
# =========================================================
with tabs[3]:
    st.subheader("📎 기획서/리포트 (Report Ver.)")

    st.markdown(
        """
## 1. 개요
**앱 이름:** Y-Compass(와이컴퍼스) — Y(연세대 노하우) + Compass(방향 잡기)  
**앱 한줄 설명:** “대학 진학이 막막한 10대에게, 근거 기반 전형/전공 후보 3개(A/B/C)와 8주 준비 로드맵을 제공하는 AI 진학 카운셀러”

### Problem Statement
- **정보 과잉/분산:** 전형·전공 정보가 흩어져 있어 무엇부터 확인해야 할지 어려움  
- **비용/접근성:** 전문 컨설팅은 비용 부담이 크고 지역·시간 제약으로 접근성이 낮음  
- **신뢰성 부족:** 경험담 중심 조언이 많아 근거·출처가 불투명

**해결 전략:** 짧은 입력 → 근거 기반 추천(출처 제시) → 즉시 실행 가능한 로드맵
"""
    )

    st.markdown("## 2. 핵심 기능(3)")
    st.markdown(
        """
1) **8문항 진학 상황 스캔(Intake & Profiling)**: 상황 요약 5줄 + 강점/제약 태그  
2) **근거 기반 후보 3개 추천(A/B/C)**: 추천 이유/액션/리스크 + 근거(출처)  
3) **8주 로드맵**: 전형+시점 반영, 주차별 목표1 + 할 일2~3 + 산출물1
"""
    )

    st.markdown("## 3. Technical Spec (피드백 반영)")
    st.table(
        [
            {
                "구분": "Input Data",
                "상세 정의": "사용자 희망 전형(직접 선택/입력) + 수시/정시 대분류 + 수시 세부 전형 분기 + 성적(내신/모의 구간) + (선택)대학/학과 키",
            },
            {
                "구분": "AI Prompting",
                "상세 정의": "전형 존중 + 가능성/리스크/대안 제시. 과거 입시 결과는 연도/범위/한계 명시, 확률 단정 금지(안정/적정/도전). 데이터 없는 경우 수치 예측 금지→전형 특성 기반 가이드.",
            },
            {
                "구분": "Output Format",
                "상세 정의": "섹션1: 가능성 카드(커버리지 배지+점수 breakdown) / 섹션2: A/B/C 점수화 차트 / 섹션3: A/B/C 카드 / 섹션4: 8주 로드맵 + 근거(expander)",
            },
        ]
    )

    st.markdown("## 4. 상용화 티어(초안)")
    st.table(
        [
            {"Tier": "Free", "제공": "진단 + A/B/C 요약 + 2주 미니 체크리스트"},
            {"Tier": "Basic", "제공": "A/B/C 상세 + 8주 로드맵 + 근거 보기"},
            {"Tier": "Pro", "제공": "전형별 심화(자소서/면접 포인트) + 리포트/PDF + 저장/버전관리(컨셉)"},
        ]
    )

    st.markdown("## 5. KPI(예시 3개)")
    st.markdown(
        """
- **Time-to-Plan**: 입력 시작→8주 플랜 생성까지 걸린 시간(분)  
- **Plan Save Rate**: 결과 저장/다운로드 비율(%)  
- **Perceived Trust**: “근거(출처) 제시가 도움이 됐다” 만족도(5점 척도)
"""
    )

st.caption("※ 본 앱은 참고용 컨설팅 도구이며, 확률 단정/합격 보장은 하지 않습니다.")
