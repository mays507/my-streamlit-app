"""
Y-Compass (와이컴스) — Streamlit MVP Prototype
- Intake (희망 전형 입력 + 수시/정시/세부전형 분기 + 성적 입력)
- Output: (1) 내가 원하는 전형 가능성 카드 (데이터 기반/가이드 기반)
          (2) AI 추천 전형/전략 TOP3 (A/B/C)
          (3) 8주 로드맵 (주차별 목표1 + 할 일 2~3 + 산출물 1)
- OpenAI Responses API optional (키 없으면 rule-based fallback)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


# =========================================================
# Page Config
# =========================================================
st.set_page_config(page_title="🧭 Y-Compass", page_icon="🧭", layout="wide")


# =========================================================
# Constants / Options
# =========================================================
ADMISSION_ROUTE = ["수시", "정시"]
SUSI_DETAIL = ["학생부교과", "학생부종합", "논술", "특기자(해당 시)"]
MAJOR_GROUPS = ["인문", "사회", "상경", "자연", "공학", "예체능", "융합/자유전공"]
ACTIVITY_PREF = ["사람(소통/리더십)", "데이터(분석/정량)", "글(에세이/스토리)", "현장(활동/프로젝트)"]
GOAL_PRIORITY = ["합격 안정성", "적성/흥미", "취업/진로 연계", "장학/비용", "지역/생활환경"]
CONSTRAINTS = ["지역(통학/거주)", "예산(비용)", "시간(병행 일정)", "가족/돌봄", "기타"]
CURRENT_STAGE = ["내신/수능 준비", "자기소개서/학생부 정리", "면접 준비", "논술 준비", "지원전략 최종 점검"]


# =========================================================
# Sample Data (MVP용: '데이터 기반' 흐름 시연)
# - 실제 서비스에서는 대학알리미/입학처 공개자료 등으로 교체
# =========================================================
# 예시: 연세대 일부 학과(샘플) 최근 3년 "교과/종합/정시" 기준선(매우 단순화)
# *주의*: 아래 값은 '샘플'이며 실제 컷/결과가 아님.
SAMPLE_ADMISSION_DATA: Dict[str, Dict[str, Dict[str, Any]]] = {
    "연세대": {
        "경영학과": {
            "학생부교과": {"years": [2022, 2023, 2024], "gpa_band": 1.5},
            "학생부종합": {"years": [2022, 2023, 2024], "gpa_band": 2.0},
            "정시": {"years": [2022, 2023, 2024], "mock_band": 1.5},
        },
        "경제학과": {
            "학생부교과": {"years": [2022, 2023, 2024], "gpa_band": 1.7},
            "학생부종합": {"years": [2022, 2023, 2024], "gpa_band": 2.2},
            "정시": {"years": [2022, 2023, 2024], "mock_band": 1.7},
        },
        "컴퓨터과학과": {
            "학생부교과": {"years": [2022, 2023, 2024], "gpa_band": 1.8},
            "학생부종합": {"years": [2022, 2023, 2024], "gpa_band": 2.3},
            "정시": {"years": [2022, 2023, 2024], "mock_band": 1.6},
        },
        "언더우드국제대학(UIC)": {
            "학생부종합": {"years": [2022, 2023, 2024], "gpa_band": 2.4},
            "정시": {"years": [2022, 2023, 2024], "mock_band": 2.0},
        },
    }
}


# =========================================================
# Utilities
# =========================================================
def _nonempty(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) and s.strip() else ""


def band_to_float(band: str) -> Optional[float]:
    """
    Convert UI band string to float threshold-ish.
    Examples:
      "1.x" -> 1.5
      "2.x" -> 2.5
      "3.x" -> 3.5
      "모름/입력안함" -> None
    """
    band = _nonempty(band)
    if not band or "모름" in band:
        return None
    if band.endswith(".x"):
        try:
            return float(band.replace(".x", "")) + 0.5
        except Exception:
            return None
    # numeric input like "2.3"
    try:
        return float(band)
    except Exception:
        return None


def classify_stability(user_value: Optional[float], ref_value: Optional[float]) -> Tuple[str, str]:
    """
    Return (label, rationale).
    Lower grade number is better. So user <= ref => 안정(높음).
    """
    if user_value is None or ref_value is None:
        return ("가이드 기반", "데이터/입력값이 충분하지 않아 수치 비교 대신 일반 전략 가이드를 제공합니다.")

    # margin: 0.3~0.5 정도를 구간으로 사용(임의, MVP 시연용)
    diff = user_value - ref_value
    if diff <= -0.2:
        return ("안정", f"입력 성적({user_value:.1f})이 기준선({ref_value:.1f})보다 유리한 편입니다.")
    if -0.2 < diff <= 0.4:
        return ("적정", f"입력 성적({user_value:.1f})이 기준선({ref_value:.1f}) 근처입니다. 전략/완성도가 중요합니다.")
    return ("도전", f"입력 성적({user_value:.1f})이 기준선({ref_value:.1f})보다 불리합니다. 대안 전형/전략 병행이 권장됩니다.")


def coverage_badge(is_data_based: bool) -> str:
    return "데이터 기반 ✅" if is_data_based else "가이드 기반 🟡"


# =========================================================
# OpenAI (Responses API)
# =========================================================
def openai_generate_plan(
    api_key: str,
    model: str,
    payload_json: Dict[str, Any],
    context_docs: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Generate structured output:
    - summary_5lines
    - routes: A/B/C each has reasons/actions/risks
    - roadmap: week1..week8 each has goal/tasks/output
    - evidence: list
    Returns dict.
    """
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    context_docs = context_docs or []
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

    r = requests.post(url, headers=headers, json=body, timeout=45)
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
        # try to salvage JSON block
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text_out[start : end + 1])
        raise


# =========================================================
# Rule-based fallback (키 없을 때도 시연 가능)
# =========================================================
def rule_based_plan(payload_json: Dict[str, Any], stability_label: str) -> Dict[str, Any]:
    major_group = payload_json.get("major_group", "전공")
    route = payload_json.get("route", "수시")
    route_detail = payload_json.get("route_detail", "")
    current_stage = payload_json.get("current_stage", "")

    summary = [
        f"현재 목표는 '{route}{(' - ' + route_detail) if route_detail else ''}' 중심의 진학 전략 수립입니다.",
        f"관심 전공군은 '{major_group}'이며, 선호 활동/강점을 바탕으로 준비 방향을 정리했습니다.",
        f"성적 입력을 기반으로 가능성은 '{stability_label}' 구간으로 분류했습니다(단정 아님).",
        f"제약 조건과 목표 우선순위를 반영해 실행 난이도와 리스크를 함께 제시합니다.",
        f"마지막으로 8주 로드맵으로 '오늘부터 할 일'로 전환합니다.",
    ]

    def mk_route(title: str) -> Dict[str, Any]:
        return {
            "title": title,
            "reasons": [
                f"{route} 준비에서 성적/활동 구조를 고려한 현실적인 선택지입니다.",
                "사용자 제약(시간/예산/지역 등)을 감안해 실행 가능성을 높였습니다.",
                "불확실성은 리스크로 분리해 관리하도록 설계했습니다.",
            ],
            "actions": [
                "전형 요건 체크리스트 작성(필수/선택 항목 분리)",
                "핵심 스토리 3개 정리(활동-역할-성과-배움)",
                "1페이지 지원전략 메모(대학/학과/전형 조합 3개)",
                "주 1회 피드백 루프(교사/선배/AI)로 수정",
                "리스크 대비용 대안 전형 1개 확보",
            ],
            "risks": [
                "데이터/정보의 최신성·정확성 한계가 있을 수 있음",
                "전형별 요구 산출물(자소서/면접/논술)이 촉박해질 수 있음",
            ],
        }

    routes = {"A": mk_route("안정"), "B": mk_route("적정"), "C": mk_route("도전")}

    # Roadmap: week1..8 goal + tasks + deliverable
    roadmap = []
    for w in range(1, 9):
        if route == "정시":
            goal = "실전 점수 안정화" if w <= 3 else ("약점 보완 집중" if w <= 6 else "실전 루틴 고정")
            tasks = [
                "모의고사/기출 1회분 풀기",
                "오답 노트 30분(원인 분류)",
                "취약 과목/단원 1개 집중 보완",
            ]
            deliverable = f"Week {w}: 오답 분류표 + 취약 단원 체크리스트"
        else:
            goal = "지원전략 확정" if w <= 2 else ("자소서/활동 정리" if w <= 5 else "면접/논술 대비")
            tasks = [
                "전형 요강 체크 + 제출물 목록화",
                "활동 3개를 STAR로 정리",
                "자소서/면접 질문 5개 초안 작성",
            ]
            deliverable = f"Week {w}: {route_detail or '수시'} 산출물 초안 1종"
        roadmap.append({"week": w, "goal": f"{goal} (현재 단계: {current_stage})", "tasks": tasks, "deliverable": deliverable})

    evidence = [
        {"title": "샘플 가이드", "note": "MVP 시연용 룰베이스 출력(실데이터 미적용)"},
    ]

    return {"summary_5lines": summary, "routes": routes, "roadmap": roadmap, "evidence": evidence}


# =========================================================
# UI — Header / Sidebar
# =========================================================
st.title("🧭 Y-Compass (와이컴퍼스)")
st.caption("연세대 AX 캠프 Track 1 — 소그룹 챌린지 | AI 진학 카운셀러 MVP")

with st.sidebar:
    st.header("🔑 API / 옵션")
    openai_key_default = st.secrets.get("OPENAI_API_KEY", "")
    openai_api_key = st.text_input("OpenAI API Key", value=openai_key_default, type="password", placeholder="sk-...")

    openai_model = st.text_input("OpenAI 모델", value="gpt-4.1-mini")
    today = st.date_input("현재 시점(로ड맵 기준)", value=date.today())

    st.divider()
    st.subheader("ℹ️ 출력 정책")
    st.write("- 확률 단정 금지 → 안정/적정/도전 구간")
    st.write("- 데이터 범위 밖은 '가이드 기반'으로 표시")

tabs = st.tabs(["📝 진단 입력", "📌 결과", "📎 기획서(요약)", "🗃️ 데이터(샘플)"])


# =========================================================
# Session State
# =========================================================
if "y_result" not in st.session_state:
    st.session_state.y_result = None
if "y_payload" not in st.session_state:
    st.session_state.y_payload = None
if "y_stability" not in st.session_state:
    st.session_state.y_stability = None
if "y_coverage" not in st.session_state:
    st.session_state.y_coverage = None


# =========================================================
# Tab 1: Intake
# =========================================================
with tabs[0]:
    st.subheader("📝 3분 진단(입력)")
    st.write("희망 전형을 **직접 선택/입력**하고, 성적(내신/모의)과 선호를 바탕으로 **A/B/C + 8주 로드맵**을 생성합니다.")

    with st.form("intake_form", clear_on_submit=False):
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("#### 1) 기본 정보")
            grade_status = st.selectbox("학년/상태", ["고3", "N수(재수/삼수)", "고2(미리보기)"])
            route = st.selectbox("희망 전형(대분류)", ADMISSION_ROUTE)
            route_detail = ""
            if route == "수시":
                route_detail = st.selectbox("수시 세부 전형", SUSI_DETAIL)

            desired_program = st.text_input("희망 전형/대학/학과(자유 입력, 선택)", placeholder="예: 연세대 경영학과 학생부종합")
            major_group = st.selectbox("관심 전공군", MAJOR_GROUPS)

            st.markdown("#### 2) 성적 정보(구간 입력)")
            gpa_band = st.selectbox("내신 등급(구간)", ["모름/입력안함", "1.x", "2.x", "3.x", "4.x", "5.x", "직접입력(예: 2.3)"])
            gpa_direct = ""
            if gpa_band.startswith("직접"):
                gpa_direct = st.text_input("내신 등급(직접)", placeholder="예: 2.3")

            mock_band = st.selectbox("모의고사 성적(구간)", ["모름/입력안함", "1.x", "2.x", "3.x", "4.x", "5.x", "직접입력(예: 2.1)"])
            mock_direct = ""
            if mock_band.startswith("직접"):
                mock_direct = st.text_input("모의고사 등급/환산(직접)", placeholder="예: 2.1")

        with c2:
            st.markdown("#### 3) 선호/비교과/제약")
            activity_pref = st.multiselect("선호 활동/강점(복수 선택)", ACTIVITY_PREF, default=[ACTIVITY_PREF[0]])
            extracurricular = st.select_slider("비교과/활동 강도(자가평가)", options=["낮음", "보통", "높음"], value="보통")
            priorities = st.multiselect("목표 우선순위(최대 2개)", GOAL_PRIORITY, default=[GOAL_PRIORITY[0], GOAL_PRIORITY[1]])
            constraints = st.multiselect("제약 조건(해당 시)", CONSTRAINTS, default=[])

            current_stage = st.selectbox("현재 단계(로ड맵 기준)", CURRENT_STAGE)
            notes = st.text_area("추가 메모(선택)", placeholder="예: 논술 병행 고려 / 면접이 특히 불안 / 통학 제약 있음")

            st.info("개인정보(학교명/실명/연락처 등) 입력은 피해주세요. 추천은 참고용이며, 최종 결정은 사용자에게 있습니다.")

        submitted = st.form_submit_button("결과 생성", type="primary")

    if submitted:
        # Normalize bands
        gpa_val = band_to_float(gpa_direct if gpa_band.startswith("직접") else gpa_band)
        mock_val = band_to_float(mock_direct if mock_band.startswith("직접") else mock_band)

        payload = {
            "today": str(today),
            "grade_status": grade_status,
            "route": route,
            "route_detail": route_detail,
            "desired_program_text": desired_program,
            "major_group": major_group,
            "gpa_band_value": gpa_val,
            "mock_band_value": mock_val,
            "activity_pref": activity_pref,
            "extracurricular_level": extracurricular,
            "priorities": priorities[:2],
            "constraints": constraints,
            "current_stage": current_stage,
            "notes": notes,
        }

        # Determine data coverage (샘플 데이터 매칭: "연세대" + 학과명 포함)
        is_data_based = False
        ref = None
        uni = "연세대"
        if _nonempty(desired_program) and uni in desired_program:
            # find major key included in text
            for major in SAMPLE_ADMISSION_DATA.get(uni, {}).keys():
                if major in desired_program:
                    # map route to key
                    key = "정시" if route == "정시" else (route_detail or "학생부종합")
                    ref = SAMPLE_ADMISSION_DATA[uni][major].get(key)
                    if ref:
                        is_data_based = True
                    break

        # stability classification
        if route == "정시":
            stability_label, rationale = classify_stability(mock_val, ref.get("mock_band") if ref else None)
        else:
            stability_label, rationale = classify_stability(gpa_val, ref.get("gpa_band") if ref else None)

        st.session_state.y_payload = payload
        st.session_state.y_stability = {"label": stability_label, "rationale": rationale}
        st.session_state.y_coverage = {"is_data_based": is_data_based, "ref": ref}

        # Build minimal context docs (MVP: 샘플 데이터 요약을 근거로 제공)
        context_docs = []
        if is_data_based and ref:
            context_docs.append(
                {
                    "title": "샘플 입시 데이터(시연용)",
                    "note": f"{uni} / {desired_program} / {route}{(' - ' + route_detail) if route_detail else ''} | "
                    f"years={ref.get('years')} | "
                    f"ref_gpa_band={ref.get('gpa_band')} ref_mock_band={ref.get('mock_band')}",
                }
            )
        else:
            context_docs.append(
                {
                    "title": "일반 전략 가이드(데이터 미보유)",
                    "note": "데이터 커버리지 밖에서는 전형 특성 기반 준비 전략/리스크/대안 중심으로 안내",
                }
            )

        # Generate result
        with st.spinner("결과를 생성하는 중..."):
            try:
                if _nonempty(openai_api_key):
                    plan = openai_generate_plan(
                        api_key=openai_api_key.strip(),
                        model=openai_model.strip(),
                        payload_json={**payload, "stability_label": stability_label, "coverage": coverage_badge(is_data_based)},
                        context_docs=context_docs,
                    )
                else:
                    plan = rule_based_plan(payload, stability_label)

                # attach evidence policy
                plan["_meta"] = {"coverage": coverage_badge(is_data_based), "stability_rationale": rationale}
                st.session_state.y_result = plan
                st.success("완료! 상단 '📌 결과' 탭에서 확인해줘.")
            except Exception as e:
                st.session_state.y_result = None
                st.error("결과 생성에 실패했어. (API 키/모델/네트워크/형식) 확인해줘.")
                st.caption(str(e))


# =========================================================
# Tab 2: Results
# =========================================================
with tabs[1]:
    st.subheader("📌 결과")
    if not st.session_state.y_result or not st.session_state.y_payload:
        st.info("먼저 '📝 진단 입력'에서 결과를 생성해줘.")
    else:
        payload = st.session_state.y_payload
        plan = st.session_state.y_result
        meta = plan.get("_meta", {})
        stability = st.session_state.y_stability or {}
        coverage = st.session_state.y_coverage or {}

        # Section 1: Possibility card
        st.markdown("### 1) 내가 원하는 전형 가능성 카드")
        c1, c2, c3 = st.columns([1.2, 1.2, 2.2], gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("**커버리지**")
                st.write(meta.get("coverage", "가이드 기반 🟡"))
        with c2:
            with st.container(border=True):
                st.markdown("**구간(안정/적정/도전)**")
                st.write(f"**{stability.get('label', '가이드 기반')}**")
        with c3:
            with st.container(border=True):
                st.markdown("**근거/설명**")
                st.write(stability.get("rationale", ""))

        st.divider()

        # Section 2: A/B/C cards
        st.markdown("### 2) AI 추천 전형/전략 TOP3 (A/B/C)")
        routes = plan.get("routes", {})
        cols = st.columns(3, gap="large")
        order = ["A", "B", "C"]
        title_map = {"A": "A (안정)", "B": "B (적정)", "C": "C (도전)"}

        for i, k in enumerate(order):
            r = routes.get(k, {})
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"#### {title_map.get(k, k)}")
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

        # Section 3: Roadmap
        st.markdown("### 3) 8주 로드맵")
        roadmap = plan.get("roadmap", [])
        if not roadmap:
            st.warning("로드맵 데이터가 비어있습니다.")
        else:
            for item in roadmap[:8]:
                w = item.get("week")
                with st.expander(f"Week {w}: {item.get('goal', '')}", expanded=(w == 1)):
                    tasks = item.get("tasks") or []
                    st.markdown("**할 일 (2~3)**")
                    for t in tasks[:3]:
                        st.write(f"- {t}")
                    st.markdown("**산출물**")
                    st.write(item.get("deliverable", ""))

        st.divider()

        # Evidence
        st.markdown("### 근거 보기(출처)")
        evidence = plan.get("evidence", [])
        if evidence:
            for ev in evidence[:10]:
                with st.expander(ev.get("title", "근거")):
                    st.write(ev.get("note", ""))
        else:
            st.caption("표시할 근거가 없습니다.")

        # Download report
        st.divider()
        st.markdown("### 결과 저장")
        report_txt = {
            "payload": payload,
            "summary_5lines": plan.get("summary_5lines", []),
            "routes": plan.get("routes", {}),
            "roadmap": plan.get("roadmap", []),
            "evidence": plan.get("evidence", []),
            "meta": meta,
        }
        st.download_button(
            "📄 결과 리포트 다운로드(.txt)",
            data=json.dumps(report_txt, ensure_ascii=False, indent=2),
            file_name="y_compass_report.txt",
            mime="text/plain",
        )


# =========================================================
# Tab 3: Proposal Summary
# =========================================================
with tabs[2]:
    st.subheader("📎 기획서(요약)")
    st.markdown(
        """
**앱 한줄 설명**  
대학 진학이 막막한 10대(고3·N수생)에게, 근거 기반 전형/전공 후보 3개(A/B/C)와 8주 준비 로드맵을 제공하는 AI 진학 카운셀러

**핵심 포인트**
- 희망 전형 직접 입력/선택 + 수시/정시 및 수시 세부 전형 분기
- 성적(내신/모의) 입력을 활용해 안정/적정/도전 구간 제시(단정 금지)
- 데이터 커버리지 내에서는 근거 제시(데이터 기반), 밖에서는 전략 가이드(가이드 기반)
- 로드맵은 전형/현재 시점 기반으로 주차별 목표/할 일/산출물 구조 고정
"""
    )

    st.markdown("#### Technical Spec (요약)")
    st.table(
        [
            {
                "구분": "Input Data",
                "상세 정의": "희망 전형 직접 입력/선택 + 수시/정시 + 수시 세부 전형 분기 + 내신/모의 성적(구간)",
            },
            {
                "구분": "AI Prompting",
                "상세 정의": "전형 존중 + 가능성/리스크/대안 제시. 근거 문서 밖 사실 단정 금지. 안정/적정/도전 구간 표현. 8주 로드맵은 주차별 목표1+할일2~3+산출물1.",
            },
            {
                "구분": "Output Format",
                "상세 정의": "1) 내가 원하는 전형 가능성 카드(커버리지 표시) 2) A/B/C TOP3 3) 8주 로드맵 + 근거(출처)",
            },
        ]
    )


# =========================================================
# Tab 4: Sample Data
# =========================================================
with tabs[3]:
    st.subheader("🗃️ 데이터(샘플)")
    st.warning(
        "아래는 MVP 시연용 '샘플 데이터'입니다. 실제 서비스에서는 대학알리미/입학처 공개 데이터 등 신뢰 가능한 출처로 교체해야 합니다."
    )
    st.json(SAMPLE_ADMISSION_DATA)

    st.markdown("#### 커버리지 동작 방식")
    st.markdown(
        """
- 사용자가 입력한 희망 전형 텍스트에 **'연세대' + 샘플 학과명**이 포함되면 → **데이터 기반**으로 표시  
- 그 외에는 → **가이드 기반**(전형 특성/전략 중심)으로 표시  
"""
    )

st.caption("© Y-Compass MVP — AX Camp Track 1 | 추천은 참고용이며, 최종 결정은 사용자에게 있습니다.")
