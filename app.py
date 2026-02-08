"""
Y-Compass (와이컴퍼스) — Streamlit MVP+ (심사자 설득력 강화 버전)

핵심 고도화(요구사항 반영)
1) CSV 업로드 기반 데이터 커버리지 확장:
   - 대학/학과/전형/연도/기준선(내신/모의) 데이터를 업로드하면
   - 자동으로 근거 컨텍스트(evidence) 생성 + 가능성 카드/그래프에 반영

2) "데이터 기반"일 때 연도별 미니 표/그래프 시각화:
   - 사용자 성적(내신/모의) vs 연도별 기준선(업로드 데이터)
   - 데이터 범위 밖이면 "가이드 기반"으로 전환

3) A/B/C 추천의 "설명가능성" 강화:
   - 성적/비교과/제약/목표 우선순위를 가중치 점수로 산출
   - 안정/적정/도전 구간 + 점수 근거(기여도 breakdown) 표시

OpenAI API(선택):
- 키가 있으면 Responses API로 A/B/C + 8주 로드맵 생성(근거 문서 기반)
- 키가 없으면 rule-based로 동작(데모 가능)

CSV 템플릿(권장 컬럼)
- university (예: 연세대)
- major (예: 경영학과)
- route (수시/정시)
- route_detail (수시일 때: 학생부교과/학생부종합/논술/특기자 등, 정시는 비워도 됨)
- year (예: 2022)
- metric (gpa 또는 mock)  # 기준선 유형: 내신(gpa) / 모의(mock)
- threshold (예: 1.7)     # 기준선(낮을수록 좋다고 가정: 등급)
- source (예: 대학알리미/입학처)
- note (선택)

⚠️ 이 앱은 "확률 단정" 금지. 안정/적정/도전 구간으로만 표현.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# =========================================================
# Page Config
# =========================================================
st.set_page_config(page_title="🧭 Y-Compass", page_icon="🧭", layout="wide")


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


# =========================================================
# Utilities
# =========================================================
def _nonempty(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) and s.strip() else ""


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


def coverage_badge(is_data_based: bool) -> str:
    return "데이터 기반 ✅" if is_data_based else "가이드 기반 🟡"


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


# =========================================================
# Data Handling: CSV -> normalized dataframe
# =========================================================
REQUIRED_COLS = ["university", "major", "route", "year", "metric", "threshold"]
OPTIONAL_COLS = ["route_detail", "source", "note"]


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # lower columns
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필수 컬럼이 누락됨: {missing}")

    # ensure optionals exist
    for c in OPTIONAL_COLS:
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

    df = df.dropna(subset=["year", "threshold"])
    df = df[df["metric"].isin(["gpa", "mock"])]

    # Clean route: accept variations
    df["route"] = df["route"].replace({"수시 ": "수시", "정시 ": "정시"})
    df = df[df["route"].isin(["수시", "정시"])]

    return df


def match_rows(
    df: pd.DataFrame,
    university: str,
    major_text: str,
    route: str,
    route_detail: str,
    metric: str,
    max_rows: int = 50,
) -> pd.DataFrame:
    """
    fuzzy-ish match:
    - exact university equals
    - major substring match (either direction)
    - route match
    - for susi: route_detail matches if provided in df; if df route_detail empty, allow fallback
    """
    if df is None or df.empty:
        return df

    uni = _nonempty(university)
    mj = _nonempty(major_text)

    sub = df[df["university"] == uni]
    if mj:
        sub = sub[
            sub["major"].apply(lambda x: (mj in str(x)) or (str(x) in mj))
        ]

    sub = sub[sub["route"] == route]
    sub = sub[sub["metric"] == metric]

    if route == "수시":
        rd = _nonempty(route_detail)
        if rd:
            # prioritize exact route_detail matches, but allow blank route_detail as generic susi row
            exact = sub[sub["route_detail"] == rd]
            generic = sub[sub["route_detail"] == ""]
            sub = pd.concat([exact, generic], ignore_index=True).drop_duplicates()

    sub = sub.sort_values("year", ascending=True).head(max_rows)
    return sub


# =========================================================
# Explainable Scoring: Why A/B/C?
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
    # each constraint adds penalty; cap
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
    """
    crude but explainable:
    - 학생부종합: 글/현장/사람 가산
    - 학생부교과: 데이터/성적 중심 가산
    - 논술: 글/데이터(논리) 가산
    - 정시: 데이터/학습 루틴 가산
    """
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

    return max(0.0, min(s, 100.0))


def academics_score(user_value: Optional[float], ref_series: Optional[pd.Series]) -> Tuple[float, str]:
    """
    Score 0..100. Lower grade is better.
    If ref not available: return neutral + guidance.
    If available: compare to median threshold (or last year).
    """
    if user_value is None:
        return 50.0, "성적 입력이 없어 학업 점수는 중립(50)으로 처리했습니다."
    if ref_series is None or ref_series.empty:
        # no data coverage -> cannot score with reference
        # still score based on absolute roughness (1.0 best .. 9.0 worst)
        s = 100.0 - (user_value - 1.0) * 15.0
        return max(10.0, min(s, 90.0)), "데이터 커버리지 밖이라 절대값 기반(거친) 점수로 처리했습니다."

    # use last year's threshold as anchor (most recent)
    anchor = float(ref_series.dropna().iloc[-1])
    diff = user_value - anchor

    # diff <= -0.2 very favorable; diff around 0 ~ 0.4 ok; diff > 0.4 hard
    if diff <= -0.2:
        s = 90.0
        msg = f"입력 성적({user_value:.1f})이 최근 기준선({anchor:.1f})보다 유리 → 학업 점수↑"
    elif diff <= 0.4:
        s = 65.0
        msg = f"입력 성적({user_value:.1f})이 최근 기준선({anchor:.1f}) 근처 → 학업 점수 중간"
    else:
        s = 35.0
        msg = f"입력 성적({user_value:.1f})이 최근 기준선({anchor:.1f})보다 불리 → 학업 점수↓"
    return s, msg


def total_score(
    w: ScoreWeights,
    acad: float,
    extra: float,
    penalty: float,
    fit: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Combine into 0..100 score with clear breakdown.
    constraints penalty subtracts.
    """
    # normalize weights to sum=1
    s = w.academics + w.extracurricular + w.constraints + w.preference_fit
    if s <= 0:
        w_norm = ScoreWeights(0.4, 0.25, 0.2, 0.15)
        s = 1.0
    else:
        w_norm = ScoreWeights(
            w.academics / s,
            w.extracurricular / s,
            w.constraints / s,
            w.preference_fit / s,
        )

    contrib_acad = acad * w_norm.academics
    contrib_extra = extra * w_norm.extracurricular
    contrib_fit = fit * w_norm.preference_fit

    # penalty applies with weight on constraints
    contrib_penalty = penalty * w_norm.constraints

    score = contrib_acad + contrib_extra + contrib_fit - contrib_penalty
    score = max(0.0, min(score, 100.0))

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


# =========================================================
# OpenAI Responses API (optional)
# =========================================================
def openai_generate_plan(
    api_key: str,
    model: str,
    payload_json: Dict[str, Any],
    context_docs: List[Dict[str, str]],
) -> Dict[str, Any]:
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
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "text": {"format": {"type": "json_object"}},
    }

    r = requests.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()

    text_out = ""
    for out_item in data.get("output", []):
        for c in out_item.get("content", []):
            if c.get("type") in ("output_text", "text") and c.get("text"):
                text_out += c["text"]
    text_out = text_out.strip()
    if not text_out:
        raise ValueError("OpenAI 응답 텍스트가 비어있습니다.")

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
        f"희망 전형은 '{route}{(' - ' + route_detail) if route_detail else ''}'이며, 입력 성적/조건 기반으로 구간은 '{band}'입니다.",
        f"관심 전공군은 '{major_group}'이고, 선호 활동 성향을 전형 특성과 매칭했습니다.",
        "데이터 커버리지 내에서는 연도/출처를 근거로 제시하고, 밖에서는 일반 전략 가이드를 제공합니다.",
        "추천은 단정이 아닌 대안 비교(A/B/C) 구조로 제공됩니다.",
        "마지막으로 8주 로드맵을 주차별 목표/할 일/산출물로 구조화합니다.",
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
# App State
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


# =========================================================
# Header
# =========================================================
st.title("🧭 Y-Compass (와이컴퍼스)")
st.caption("연세대 AX 캠프 Track 1 — 소그룹 챌린지 | 근거 기반 AI 진학 카운셀러 (MVP+)")


# =========================================================
# Sidebar: API + Weights + Pricing (심사자 관점)
# =========================================================
with st.sidebar:
    st.header("🔑 OpenAI (선택)")
    openai_key_default = st.secrets.get("OPENAI_API_KEY", "")
    openai_api_key = st.text_input("OpenAI API Key", value=openai_key_default, type="password", placeholder="sk-...")
    openai_model = st.text_input("모델", value="gpt-4.1-mini")

    st.divider()
    st.header("⚙️ 점수 가중치(설명가능성)")
    st.caption("A/B/C 구간은 점수로 산출되며, 각 요소의 기여도를 공개합니다.")
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
    st.write("여기 업로드된 데이터가 **근거(출처/연도) + 가능성 그래프 + 점수 산정**에 반영됩니다.")

    with st.expander("CSV 템플릿(권장) 보기", expanded=True):
        st.code(
            "university,major,route,route_detail,year,metric,threshold,source,note\n"
            "연세대,경영학과,수시,학생부종합,2022,gpa,2.0,입학처,예시\n"
            "연세대,경영학과,수시,학생부종합,2023,gpa,2.1,입학처,예시\n"
            "연세대,경영학과,정시,,2024,mock,1.6,입학처,예시\n",
            language="text",
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
            rd = st.selectbox("수시 세부", sorted(df[(df["university"] == u) & (df["major"] == m) & (df["route"] == "수시")]["route_detail"].unique().tolist()))
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
        # parse numeric
        gpa_val = band_to_float(gpa_direct if gpa_band == "직접입력" else gpa_band)
        mock_val = band_to_float(mock_direct if mock_band == "직접입력" else mock_band)

        # pick metric by route
        metric = "mock" if route == "정시" else "gpa"
        user_metric_value = mock_val if metric == "mock" else gpa_val

        university = _nonempty(desired_university) or (_nonempty(desired_text).split()[0] if _nonempty(desired_text) else "")
        major = _nonempty(desired_major) or ""

        # match data rows if available
        matched = pd.DataFrame()
        if use_df and university and major:
            matched = match_rows(df_all, university, major, route, route_detail, metric)

        is_data_based = (matched is not None) and (not matched.empty)

        # academics score uses ref series if data-based
        ref_series = matched.sort_values("year")["threshold"] if is_data_based else None
        acad_s, acad_msg = academics_score(user_metric_value, ref_series)

        extra_s = extracurricular_score(extracurricular)
        penalty = constraints_penalty(constraints)
        fit_s = preference_fit_score(activity_pref, route, route_detail)

        tot, breakdown = total_score(weights, acad_s, extra_s, penalty, fit_s)
        band = score_to_band(tot)

        payload = {
            "today": str(today),
            "grade_status": grade_status,
            "route": route,
            "route_detail": route_detail,
            "desired_university": university,
            "desired_major": major,
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
            "coverage": coverage_badge(is_data_based),
            "score_total": float(tot),
            "score_breakdown": breakdown,
            "scoring_notes": {
                "academics": acad_msg,
                "fit": "전형-성향 적합도는 선택 성향과 전형 특성 매칭으로 산출했습니다.",
                "constraints": "제약은 감점으로 적용되며(가중치 반영), 많을수록 리스크가 커집니다.",
            },
        }
        st.session_state.payload = payload
        st.session_state.score_breakdown = breakdown

        # Build evidence (RAG context)
        evidence_docs: List[Dict[str, str]] = []
        if is_data_based:
            # compact the matched rows as evidence
            for _, row in matched.sort_values("year").tail(10).iterrows():
                title = f"{row['university']} {row['major']} | {row['route']}{(' - ' + row['route_detail']) if row['route_detail'] else ''} | {int(row['year'])}"
                note = f"metric={row['metric']} threshold={row['threshold']} | source={row.get('source','')} | note={row.get('note','')}"
                evidence_docs.append({"title": title, "note": note})
        else:
            evidence_docs.append(
                {
                    "title": "데이터 미보유(가이드 기반)",
                    "note": "해당 대학/학과/전형의 업로드 데이터가 없어 수치 기반 예측을 제공하지 않고, 전형 특성 기반 전략 가이드로 안내합니다.",
                }
            )

        st.session_state.evidence = evidence_docs

        # Generate plan (OpenAI if key else rule-based)
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
                    "coverage": payload["coverage"],
                    "score_total": payload["score_total"],
                    "score_breakdown": payload["score_breakdown"],
                    "academics_msg": acad_msg,
                }
                st.session_state.result = plan
                st.success("완료! '📌 결과' 탭에서 확인해줘.")
            except Exception as e:
                st.session_state.result = None
                st.error("생성 실패(키/모델/네트워크/JSON 형식) 확인")
                st.caption(str(e))


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

        # --- Section 1: Possibility Card (Technical Spec 그대로)
        st.markdown("## 섹션 1 — 내가 원하는 전형 가능성 카드")
        a, b, c = st.columns([1.0, 1.0, 2.2], gap="large")

        with a:
            with st.container(border=True):
                st.markdown("**커버리지**")
                st.write(payload["coverage"])
                st.caption("데이터 기반이면 연도/출처 근거 및 그래프 제공")

        with b:
            with st.container(border=True):
                st.markdown("**구간(안정/적정/도전)**")
                st.write(f"**{payload['band_label']}**")
                st.caption(f"총점: {payload['score_total']:.1f} / 100")

        with c:
            with st.container(border=True):
                st.markdown("**왜 이 구간인가? (설명가능 점수화)**")
                st.write(meta.get("academics_msg", ""))
                st.markdown("**기여도(가중치 반영)**")
                bd = payload["score_breakdown"]
                st.write(f"- 학업(성적): {bd['학업(성적)']:.1f}")
                st.write(f"- 비교과: {bd['비교과']:.1f}")
                st.write(f"- 적합도: {bd['적합도(성향↔전형)']:.1f}")
                st.write(f"- 제약(감점): {bd['제약(감점)']:.1f}")

        st.divider()

        # --- Evidence visualization when data-based
        df_all = st.session_state.df_data if st.session_state.df_data is not None else pd.DataFrame()
        if payload["coverage"].startswith("데이터 기반") and not df_all.empty:
            st.markdown("### 근거 시각화(연도별 기준선 vs 내 성적)")
            metric = payload["metric_used"]
            university = payload.get("desired_university", "")
            major = payload.get("desired_major", "")
            route = payload.get("route", "수시")
            route_detail = payload.get("route_detail", "")

            matched = match_rows(df_all, university, major, route, route_detail, metric)
            if matched is not None and not matched.empty:
                chart_df = matched.sort_values("year")[["year", "threshold"]].copy()
                chart_df["user_value"] = payload.get("metric_value", None)

                st.dataframe(chart_df, use_container_width=True)

                st.line_chart(chart_df.set_index("year")[["threshold", "user_value"]])
                st.caption("※ 등급 기준: 낮을수록 유리. (그래프는 참고용이며, 단정적 합격 예측이 아님)")
            else:
                st.info("시각화할 매칭 데이터가 없습니다(학과명/전형 키워드 확인).")

        st.divider()

        # --- Section 2: A/B/C cards
        st.markdown("## 섹션 2 — AI 추천 전형/전략 TOP3 (A/B/C)")
        routes = plan.get("routes", {})
        cols = st.columns(3, gap="large")
        keys = ["A", "B", "C"]
        title_map = {"A": "A: 안정", "B": "B: 적정", "C": "C: 도전"}

        for i, k in enumerate(keys):
            r = routes.get(k, {})
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"### {title_map[k]}")
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

        # --- Section 3: 8-week Roadmap
        st.markdown("## 섹션 3 — 8주 로드맵")
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

        # --- Evidence (RAG)
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
# Tab 4: Report / Spec (보고서 느낌 + Technical Spec 반영)
# =========================================================
with tabs[3]:
    st.subheader("📎 기획서/리포트 (Report Ver.)")

    st.markdown(
        """
## 1. 개요
**앱 이름:** Y-Compass(와이컴퍼스) — Y(연세대 노하우) + Compass(방향 잡기)  
**앱 한줄 설명:** 대학 진학이 막막한 10대에게, 근거 기반 후보 3개(A/B/C)와 8주 로드맵을 제공하는 AI 진학 카운셀러

### Problem Statement
- **정보 과잉/분산:** 전형·전공 정보가 흩어져 있어 무엇부터 볼지 어려움  
- **비용/접근성:** 컨설팅은 비싸고 지역/시간 제약이 큼  
- **신뢰성 부족:** “카더라” 조언이 많아 근거·출처가 불투명

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
                "상세 정의": "사용자 희망 전형(직접 선택/입력) + 수시/정시 대분류 + 수시 세부 전형 분기 + 성적(내신/모의 구간)",
            },
            {
                "구분": "AI Prompting",
                "상세 정의": "전형 존중 + 가능성/리스크/대안 제시. 과거 입시 결과는 연도/범위/한계 명시, 확률 단정 금지(안정/적정/도전). 데이터 없는 경우 수치 예측 금지→전형 특성 기반 가이드.",
            },
            {
                "구분": "Output Format",
                "상세 정의": "섹션1: 내가 원하는 전형 가능성 카드(커버리지 표시) / 섹션2: TOP3(A/B/C) / 섹션3: 8주 로드맵 + 근거(expander)",
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
