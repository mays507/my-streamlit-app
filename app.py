# app.py
# =========================================================
# Y-Compass (2026 수시) — PDF 기반 전형 KB 구축(정규식+LLM 하이브리드)
#  - PDF에서 전형명/전형요소/반영비율/수능최저/일정 자동 추출 → 표로 보여줌
#  - 추출 결과를 JSON DB로 저장 → 앱 실행 시 즉시 로딩
#  - 사용자 프로필과 전형 조건 매칭 → Top3 추천 + 8주 로드맵
#
# requirements.txt 권장:
#   streamlit
#   pandas
#   pypdf
#   openai>=1.0.0
#   reportlab   # (있어도 되고 없어도 됨. 이 앱은 비의존)
#   # 선택:
#   pdfplumber  # pypdf가 텍스트 잘 못뽑는 PDF 대비(텍스트 PDF 권장)
# =========================================================

import os
import re
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Y-Compass (2026 수시) — 전형 KB + 추천 + 8주 플랜", layout="wide")
st.title("🎓 Y-Compass (2026 수시) — 전형 KB(PDF→표) + 전형 추천 + 8주 로드맵")
st.caption("서울대/연세대/고려대 모집요강 PDF를 업로드 → 전형 정보를 구조화(KB) → ‘몰라서 못 지원’ 줄이고 전략/플랜까지 자동화.")

# -----------------------------
# Paths (Streamlit Cloud에서도 파일 저장은 '세션/컨테이너' 내에 가능)
#   - 배포 환경에 따라 재시작 시 초기화될 수 있음(=DB 백업 다운로드 제공)
# -----------------------------
DATA_DIR = "data"
KB_PATH = os.path.join(DATA_DIR, "admission_kb.json")

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Optional: OpenAI
# -----------------------------
def get_openai_client(api_key: str):
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def llm_json(client, model: str, system: str, user: str) -> Optional[Any]:
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content.strip()
        # 모델이 코드펜스 붙이면 제거
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
        return json.loads(txt)
    except Exception:
        return None

# -----------------------------
# PDF extraction
# -----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    # pypdf 우선
    try:
        from pypdf import PdfReader
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        texts = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            if t.strip():
                texts.append(f"\n\n[PAGE {i+1}]\n{t}")
        joined = "\n".join(texts).strip()
        if len(joined) > 400:
            return joined
    except Exception:
        pass

    # pdfplumber fallback(설치된 경우)
    try:
        import pdfplumber
        uploaded_file.seek(0)
        texts = []
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text() or ""
                if t.strip():
                    texts.append(f"\n\n[PAGE {i+1}]\n{t}")
        return "\n".join(texts).strip()
    except Exception:
        return ""

def normalize_space(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# -----------------------------
# KB I/O
# -----------------------------
def load_kb() -> Dict[str, Any]:
    ensure_data_dir()
    if not os.path.exists(KB_PATH):
        return {"version": 1, "updated_at": None, "universities": {}}
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "updated_at": None, "universities": {}}

def save_kb(kb: Dict[str, Any]) -> None:
    ensure_data_dir()
    kb["updated_at"] = int(time.time())
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

# -----------------------------
# Heuristic parsing (Regex-first)
#   목표: 전형명/요소/반영비율/수능최저/일정
# -----------------------------
TRACK_NAME_PATTERNS = [
    r"학생부\s*종합(?:\s*\[[^\]]+\])?",
    r"학생부\s*교과(?:\s*\[[^\]]+\])?",
    r"논술\s*전형",
    r"실기(?:/실적)?\s*전형",
    r"특기자\s*전형",
    r"기회\s*균형(?:\s*전형)?",
    r"고른\s*기회(?:\s*전형)?",
    r"지역\s*균형(?:\s*전형)?",
    r"일반\s*전형",
    r"학교\s*추천(?:\s*전형)?",
    r"학업\s*우수(?:\s*전형)?",
    r"계열\s*적합(?:\s*전형)?",
    r"국제\s*형(?:\s*전형)?",
    r"추천\s*형(?:\s*전형)?",
]

KEY_ELEMENT_WORDS = [
    "서류", "면접", "논술", "수능", "최저", "학생부", "교과", "비교과", "활동", "자기소개서",
    "추천", "실기", "실적", "출결", "봉사", "세특", "전공적합", "학업역량", "발전가능성"
]

def find_percent_ratios(block: str) -> List[str]:
    # 예: "교과 90% + 서류 10%", "서류 100%", "1단계: 서류 100%, 2단계: 1단계 70% + 면접 30%"
    ratios = re.findall(r"(?:서류|면접|논술|교과|학생부|실기|수능)\s*\d{1,3}\s*%|(?:\d{1,3}\s*%)", block)
    # 중복/노이즈 줄이기
    cleaned = []
    for r0 in ratios:
        r0 = r0.strip()
        if r0 not in cleaned:
            cleaned.append(r0)
    return cleaned[:12]

def find_schedule_lines(text: str) -> List[str]:
    # 일정 관련 패턴: 원서접수/서류제출/1단계발표/면접/논술/최종발표 등
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    hits = []
    for ln in lines:
        if any(k in ln for k in ["원서", "접수", "서류", "제출", "발표", "면접", "논술", "실기", "등록"]):
            if re.search(r"\d{1,2}\s*월|\d{1,2}\.\d{1,2}|\d{4}\.\d{1,2}", ln):
                hits.append(ln)
    # 너무 많으면 상위 몇 개만
    return hits[:20]

def detect_csat_min(block: str) -> str:
    # 수능최저/최저학력기준/응시영역 등
    if re.search(r"수능\s*최저|최저\s*학력|최저학력기준", block):
        # 구체 문장 일부만 추출
        m = re.search(r"(수능\s*최저[^\n\.]{0,120}|최저\s*학력[^\n\.]{0,120}|최저학력기준[^\n\.]{0,120})", block)
        return (m.group(0).strip() if m else "수능최저 있음(상세는 요강 확인)")
    if re.search(r"수능\s*최저\s*없|최저\s*없", block):
        return "수능최저 없음"
    return "요강 확인"

def extract_key_elements(block: str) -> List[str]:
    found = []
    for w in KEY_ELEMENT_WORDS:
        if w in block and w not in found:
            found.append(w)
    # “요소”는 너무 많아지면 의미 없으니 상위만
    return found[:12]

def split_into_sections(text: str) -> List[str]:
    # 전형 단위로 자르기: 전형명 후보가 등장하는 지점 기준
    # 1) 줄 단위로 스캔하며 전형명 매치되는 줄을 헤더로 간주
    lines = text.split("\n")
    header_idxs = []
    header_regex = re.compile("|".join(TRACK_NAME_PATTERNS))
    for i, ln in enumerate(lines):
        if header_regex.search(ln.replace(" ", "")) or header_regex.search(ln):
            header_idxs.append(i)

    if not header_idxs:
        # 못 찾으면 전체를 한 덩어리로
        return [text]

    header_idxs = sorted(set(header_idxs))
    sections = []
    for j, idx in enumerate(header_idxs):
        start = idx
        end = header_idxs[j+1] if j+1 < len(header_idxs) else len(lines)
        sec = "\n".join(lines[start:end]).strip()
        if len(sec) > 60:
            sections.append(sec)
    return sections[:60]  # 과도 방지

def regex_parse_routes(university: str, year: str, text: str, source_file: str) -> Dict[str, Any]:
    text = normalize_space(text)
    sections = split_into_sections(text)

    # 일정은 전체에서 뽑아 “공통 일정”으로도 저장(필요 시 전형별에 붙임)
    common_schedule = find_schedule_lines(text)

    routes = []
    header_regex = re.compile("|".join(TRACK_NAME_PATTERNS))

    for sec in sections:
        # 전형명 후보: 섹션 첫 줄/초반에서 잡기
        head = sec.split("\n")[0][:80]
        m = header_regex.search(head) or header_regex.search(sec[:200])
        if not m:
            continue
        route_name = re.sub(r"\s+", " ", m.group(0)).strip()

        # 전형요소/반영비율/수능최저
        elements = extract_key_elements(sec)
        ratios = find_percent_ratios(sec)
        csat_min = detect_csat_min(sec)

        # 섹션 내 “전형요소” 문장 일부 요약(정규식)
        # 너무 길면 잘라서 notes로
        notes = sec[:900]
        notes = re.sub(r"\s+", " ", notes).strip()

        routes.append({
            "university": university,
            "year": year,
            "route_name": route_name,
            "key_elements": elements,
            "evaluation_ratio": ratios,   # 리스트 형태(원문에서 잡힌 % 조각들)
            "csat_minimum": csat_min,
            "schedule": common_schedule[:8],  # 공통 일정 상위만
            "source": {"file": source_file},
            "confidence": "regex",
            "notes": notes
        })

    # 중복 전형명 합치기(가장 정보 많은 것 우선)
    merged = {}
    for r in routes:
        k = r["route_name"]
        if k not in merged:
            merged[k] = r
        else:
            # 정보량 비교 후 더 풍부한 쪽으로 업데이트
            def info_score(x):
                return len(x.get("key_elements", [])) + len(x.get("evaluation_ratio", [])) + len(" ".join(x.get("schedule", [])))
            if info_score(r) > info_score(merged[k]):
                merged[k] = r

    return {
        "university": university,
        "year": year,
        "source_file": source_file,
        "routes": list(merged.values()),
        "common_schedule": common_schedule,
        "parser": {"mode": "regex", "route_count": len(merged)}
    }

# -----------------------------
# LLM refine (Hybrid: regex output -> structured cleanup)
# -----------------------------
def llm_refine_routes(client, model: str, parsed: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    if client is None:
        return parsed

    system = (
        "너는 한국 대학 입학전형(수시) 모집요강을 구조화하는 데이터 엔지니어다. "
        "주어진 (1) 정규식 1차 추출 결과와 (2) 원문 발췌를 참고하여, 전형 정보를 더 정확하고 일관된 JSON으로 정리한다. "
        "정보가 불명확하면 '요강 확인'이라고 써라. 절대 환각(없는 내용 생성) 금지."
    )

    # 원문은 길 수 있으니 앞부분/키워드 주변만 제한 발췌
    excerpt = raw_text[:8000]

    user = f"""
[정규식 1차 추출 결과 JSON]
{json.dumps(parsed, ensure_ascii=False, indent=2)[:7000]}

[모집요강 원문 발췌]
{excerpt}

요청:
- routes 배열을 유지하되, 각 route에 대해 아래 필드를 정리해라.
  - route_name: 가능한 정확한 전형 공식명
  - key_elements: ["서류","면접","논술","교과"...] 등 핵심 요소만
  - evaluation_ratio: 가능하면 "서류 100%", "교과 90% + 서류 10%" 같은 문장 형태로 1~3개로 정리
  - csat_minimum: "수능최저 없음" 또는 "수능최저 있음: (요강 문장 일부)" 또는 "요강 확인"
  - schedule: 원서/서류/1단계/면접/논술/최종 발표 등 핵심 일정 3~8개
  - notes: 중복지원 제한/추천 필요/자격 요건 등 주의사항 요약 1~3줄
- JSON만 출력(코드펜스 없이).
"""

    refined = llm_json(client, model, system, user)
    if refined and isinstance(refined, dict) and "routes" in refined:
        # 표준 필드가 빠졌을 때 보완
        refined.setdefault("university", parsed.get("university"))
        refined.setdefault("year", parsed.get("year"))
        refined.setdefault("source_file", parsed.get("source_file"))
        refined.setdefault("common_schedule", parsed.get("common_schedule", []))
        refined.setdefault("parser", {"mode": "hybrid", "route_count": len(refined.get("routes", []))})
        # confidence 마크
        for r in refined.get("routes", []):
            r["confidence"] = "hybrid"
        return refined

    return parsed

# -----------------------------
# KB utilities: routes -> DataFrame
# -----------------------------
def routes_to_df(routes: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in routes:
        rows.append({
            "전형명(route_name)": r.get("route_name", ""),
            "전형요소(key_elements)": ", ".join(r.get("key_elements", []) or []),
            "반영비율(evaluation_ratio)": " | ".join(r.get("evaluation_ratio", []) or []),
            "수능최저(csat_minimum)": r.get("csat_minimum", ""),
            "일정(schedule)": " / ".join(r.get("schedule", []) or []),
            "주의사항(notes)": r.get("notes", ""),
            "confidence": r.get("confidence", "")
        })
    return pd.DataFrame(rows)

def upsert_university_kb(kb: Dict[str, Any], uni: str, year: str, payload: Dict[str, Any]) -> None:
    kb.setdefault("universities", {})
    kb["universities"].setdefault(uni, {})
    kb["universities"][uni][year] = payload

def get_university_routes(kb: Dict[str, Any], uni: str, year: str) -> List[Dict[str, Any]]:
    try:
        return kb["universities"][uni][year]["routes"]
    except Exception:
        return []

# -----------------------------
# Recommendation logic (KB 기반)
#   - KB가 없으면 fallback(간이 룰)로라도 추천
# -----------------------------
def score_route_by_profile(route: Dict[str, Any], profile: Dict[str, Any]) -> float:
    gpa = profile["gpa"]              # 1~9
    ecs = profile["ecs_strength"]     # 1~5
    interview = profile["interview"]  # 1~5
    nonsul = profile["nonsul"]        # 1~5
    intl = profile["international"]   # 1~5
    reco_ok = profile["reco_ok"]      # bool
    qualification = profile["qualification"]  # bool

    name = route.get("route_name", "")
    elems = route.get("key_elements", []) or []
    csat = route.get("csat_minimum", "")
    notes = route.get("notes", "") or ""
    ratio_text = " ".join(route.get("evaluation_ratio", []) or [])

    s = 0.0

    # 전형 타입 감지(이름/요소 기반)
    is_gyogwa = ("교과" in name) or ("학교추천" in name) or ("추천" in name and "종합" not in name)
    is_jonghap = ("종합" in name) or ("계열" in name) or ("활동" in name) or ("일반" in name)
    is_nonsul = ("논술" in name) or ("논술" in elems)

    is_reco = ("추천" in name) or ("학교장" in notes) or ("추천" in notes)
    is_gw = ("기회" in name) or ("고른기회" in name) or ("다문화" in name) or ("정원외" in notes)

    # 추천 필요 전형인데 추천 불가면 페널티
    if is_reco and not reco_ok:
        s -= 2.0
    if is_reco and reco_ok:
        s += 0.6

    # 기회/자격 전형 가산
    if is_gw and qualification:
        s += 2.0
    if is_gw and not qualification:
        s -= 0.6

    # 내신 민감도(교과/교과비중이 큰 것)
    # ratio_text에 교과 70~100% 언급 있으면 내신 민감
    gpa_sensitive = is_gyogwa or bool(re.search(r"교과\s*\d{1,3}\s*%", ratio_text))

    if gpa_sensitive:
        if gpa <= 2.5: s += 2.2
        elif gpa <= 4.0: s += 1.1
        elif gpa <= 6.0: s += 0.0
        else: s -= 1.2
    else:
        # 종합/논술은 내신 낮아도 여지
        if gpa >= 5.0 and (is_jonghap or is_nonsul):
            s += 0.6

    # 비교과/면접/논술 역량
    if is_jonghap:
        s += (ecs - 3) * 0.9
    if "면접" in elems or "면접" in ratio_text or "면접" in notes:
        s += (interview - 3) * 0.7
    if is_nonsul:
        s += (nonsul - 3) * 1.0

    # 국제형/글로벌
    if "국제" in name:
        s += (intl - 3) * 0.9

    # 수능최저 리스크(사용자 입력이 없으니, 기본적으로 “최저 있음”은 약간 페널티)
    if "수능최저 있음" in csat or "최저" in csat and "없음" not in csat:
        s -= 0.3

    return s

# -----------------------------
# 8-week roadmap (템플릿 + LLM optional)
# -----------------------------
def roadmap_template(uni: str, route_name: str) -> List[Dict[str, Any]]:
    is_nonsul = "논술" in route_name
    is_gyogwa = ("교과" in route_name) or ("학교추천" in route_name) or ("추천" in route_name and "종합" not in route_name)
    is_jonghap = ("종합" in route_name) or ("계열" in route_name) or ("활동" in route_name) or ("일반" in route_name)
    is_gw = ("기회" in route_name) or ("고른기회" in route_name)

    plan = []
    for w in range(1, 9):
        p = {"week": w, "goal": "", "tasks": []}
        if w == 1:
            p["goal"] = "요강 체크리스트 + 내 스펙 갭 분석"
            p["tasks"] = [
                f"{uni} {route_name} 요강에서 자격/제출/일정/최저 체크리스트(1p) 만들기",
                "내신/비교과/수능/면접/논술 현황을 ‘가능-리스크’로 표기(갭표)",
                "지원학과 3~5개 후보 확정 + 전형별 리스크(추천/최저/면접) 표시"
            ]
        elif w == 2:
            p["goal"] = "지원동기/전공적합 스토리 초안"
            p["tasks"] = [
                "활동 3개를 ‘문제-행동-성과-배움’ 구조로 재서술(근거 중심)",
                "전공 관련 탐구/독서/프로젝트 근거 3개 확정(링크/자료 포함)",
                "면접 대비용 1분 자기소개 + 꼬리질문 10개 세트"
            ]
        elif w == 3:
            if is_nonsul:
                p["goal"] = "논술 유형 분석 + 기출 2~3회"
                p["tasks"] = [
                    "해당 대학 논술 기출 유형/채점 포인트 1p 요약",
                    "기출 2회 ‘실전 시간’으로 작성 + 자기 첨삭 체크리스트",
                    "자주 무너지는 포인트(개요/근거/문장/시간) 5개 규칙화"
                ]
            else:
                p["goal"] = "서류/면접 실전화(증거 중심)"
                p["tasks"] = [
                    "서류 문장 ‘추상→구체’로 리라이팅(숫자/역할/결과 포함)",
                    "예상질문 20개 답변을 ‘결론-근거-예시’ 포맷으로 통일",
                    "모의면접 1회(녹화) 후 피드백 10개 반영"
                ]
        elif w == 4:
            if is_gyogwa:
                p["goal"] = "교과 강점 극대화 + 최저/추천 리스크 관리"
                p["tasks"] = [
                    "반영 과목/가중치 기준으로 강점 과목표 만들기",
                    "수능최저가 있다면 4주 단기 최저 플랜(영역별 목표) 설정",
                    "추천 필요 시 담임/진학부 컨택 일정 확정(추천 가능성 체크)"
                ]
            else:
                p["goal"] = "맞춤형 지원 전략 고도화"
                p["tasks"] = [
                    "학과별 ‘왜 여기?’ 문장 3개씩(근거 포함) 제작",
                    "포트폴리오/증빙 자료 폴더 정리(파일명 규칙 통일)",
                    "모의면접 1회(또는 논술 1회) 추가 + 약점 보완 계획"
                ]
        elif w == 5:
            p["goal"] = "수시 6장 조합 확정(플랜A/B)"
            p["tasks"] = [
                "상향/적정/안정 밸런스표 작성(각 전형의 합격 포인트 3줄)",
                "가장 약한 구간 1개를 선정해 2주 집중 보완(면접/논술/최저)",
                "제출물/서류 작업 일정표(마감 역산) 완성"
            ]
        elif w == 6:
            if is_nonsul:
                p["goal"] = "논술 실전 주간(모의 3회)"
                p["tasks"] = [
                    "기출 2회 + 모의 1회(실전 시간/환경) 수행",
                    "첨삭 체크: 논지 일관/근거 질/문장 명료/시간 배분",
                    "금지 패턴(5개) 확정 + 최종 템플릿(개요 구조) 완성"
                ]
            else:
                p["goal"] = "면접/서류 최종 압축"
                p["tasks"] = [
                    "빈출 질문(동기/전공/협업/갈등/성장) 스크립트 최종본",
                    "내 활동의 ‘객관적 표현’ 10문장(역할/성과/지표) 정리",
                    "모의면접 2회(또는 질의응답 30문항)로 안정화"
                ]
        elif w == 7:
            p["goal"] = "제출/원서 실수 방지 체크 완료"
            p["tasks"] = [
                "파일/양식/글자수/개인정보/증빙 누락 최종 점검표 완료",
                "추천/학교 제출(해당 시) 마감 전 완료(증빙 캡처/확인)",
                "최종본 백업(클라우드+로컬) + 파일명 규칙 확정"
            ]
        elif w == 8:
            p["goal"] = "실전 컨디션 세팅 + 플랜B(정시/추가합격) 준비"
            p["tasks"] = [
                "시험/면접 당일 루틴(수면/식사/이동/준비물) 체크리스트 출력",
                "대학별 10분 요약 노트(전공/활동/질문) 제작",
                "결과 대기 플랜: 추가합격 대응 + 정시 전환 체크리스트"
            ]

        if is_gw:
            p["tasks"].append("※ (자격 전형) 자격/증빙 누락이 즉탈 포인트 → 증빙 체크를 항상 1순위로 고정")
        if is_jonghap:
            p["tasks"].append("※ (학종) ‘활동 나열’ 금지: 모든 문장을 근거/역할/결과로 증명")

        plan.append(p)
    return plan

def roadmap_llm_refine(client, model: str, uni: str, route: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = roadmap_template(uni, route.get("route_name", ""))
    if client is None:
        return base

    system = "너는 한국 입시 코치다. 전형 조건과 사용자 프로필을 반영해 8주 로드맵을 매우 구체적 산출물 중심으로 작성한다."
    user = f"""
대학: {uni}
전형: {route.get('route_name')}
전형 정보:
- 전형요소: {route.get('key_elements')}
- 반영비율: {route.get('evaluation_ratio')}
- 수능최저: {route.get('csat_minimum')}
- 일정: {route.get('schedule')}
- 주의사항: {route.get('notes')}

사용자 프로필:
{json.dumps(profile, ensure_ascii=False, indent=2)}

요청:
- 8주 로드맵을 JSON 배열로만 출력하라.
- 각 원소: {{"week":1,"goal":"...","tasks":["..."]}}
- tasks는 “산출물 중심” (체크리스트/스크립트/기출 n회/폴더정리 등)
"""
    refined = llm_json(client, model, system, user)
    if isinstance(refined, list) and len(refined) == 8:
        return refined
    return base

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("OpenAI API Key (선택)", type="password", help="넣으면: PDF 파싱 정교화(2차) + 8주 플랜 문장력이 확 올라가요.")
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    st.divider()
    st.subheader("📌 대학/연도")
    uni = st.selectbox("대학", ["서울대", "연세대", "고려대"], index=1)
    year = st.selectbox("연도", ["2026"], index=0)

client = get_openai_client(api_key) if api_key else None

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📚 전형 정보(KB) 구축", "🧭 전형 추천 + 8주 로드맵", "🗂️ DB 관리/내보내기"])

# =========================================================
# TAB 1: KB 구축 (PDF -> regex parse -> optional LLM refine -> save JSON)
# =========================================================
with tab1:
    st.subheader("📚 전형 정보(KB) — 모집요강 PDF 업로드 → 전형표 자동 생성(정규식+LLM)")
    st.write(
        "너가 말한 핵심(“몰라서 지원 못하는 사람”)을 해결하는 레이어야. "
        "PDF에서 전형 정보를 구조화해서 저장해두면, 이후 추천/설명이 **근거 기반**으로 돌아가."
    )

    kb = load_kb()
    uploaded_files = st.file_uploader(
        "서울대/연세대/고려대 2026 수시 모집요강 PDF 업로드 (여러 개 가능)",
        type=["pdf"],
        accept_multiple_files=True
    )

    colA, colB = st.columns([1, 1])
    with colA:
        use_llm_refine = st.checkbox("LLM으로 파싱 결과 정교화(권장)", value=bool(api_key))
        st.caption("정규식 1차 → LLM 2차 보정(환각 금지 프롬프트)로 ‘반영비율/최저/주의사항’이 더 깔끔해져요.")

    with colB:
        st.info("⚠️ 스캔본(이미지) PDF는 텍스트 추출이 잘 안 될 수 있어요. (초기 MVP는 텍스트 PDF 우선 지원)")

    if st.button("📌 업로드 PDF 분석 → KB 저장", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("PDF를 업로드해줘.")
        else:
            all_routes_preview = []
            with st.spinner("PDF 텍스트 추출 + 전형 파싱 중..."):
                for f in uploaded_files:
                    raw = extract_text_from_pdf(f)
                    if not raw or len(raw) < 200:
                        st.warning(f"텍스트 추출 실패/부족: {f.name} (스캔본일 가능성)")
                        continue

                    parsed = regex_parse_routes(uni, year, raw, source_file=f.name)

                    if use_llm_refine and client is not None:
                        parsed = llm_refine_routes(client, model, parsed, raw)

                    # KB upsert
                    upsert_university_kb(kb, uni, year, parsed)

                    # preview
                    for r in parsed.get("routes", []):
                        all_routes_preview.append(r)

            save_kb(kb)
            st.success(f"KB 저장 완료 ✅  ({uni} {year}) — 전형 {len(all_routes_preview)}개 파싱/저장")

    # Show current KB summary
    kb = load_kb()
    routes = get_university_routes(kb, uni, year)
    st.markdown("### ✅ 현재 저장된 전형 표")
    if routes:
        df = routes_to_df(routes)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("※ confidence=hybrid면 ‘정규식+LLM 보정’이 적용된 상태.")
    else:
        st.warning("아직 KB가 비어있어. 위에서 PDF 분석→KB 저장을 먼저 해줘.")

    # Allow manual edit (light)
    st.markdown("### ✍️ (선택) 전형 데이터 수동 보정")
    st.caption("PDF 파싱이 애매한 경우, 여기서 전형명/비율/최저/주의사항을 직접 수정해도 돼.")
    if routes:
        edit_df = routes_to_df(routes).copy()
        edited = st.data_editor(edit_df, use_container_width=True, num_rows="dynamic")
        if st.button("💾 수동 수정 저장", use_container_width=True):
            # edited DF -> routes back (간단 매핑)
            new_routes = []
            for _, row in edited.iterrows():
                new_routes.append({
                    "university": uni,
                    "year": year,
                    "route_name": str(row.get("전형명(route_name)", "")).strip(),
                    "key_elements": [x.strip() for x in str(row.get("전형요소(key_elements)", "")).split(",") if x.strip()],
                    "evaluation_ratio": [x.strip() for x in str(row.get("반영비율(evaluation_ratio)", "")).split("|") if x.strip()],
                    "csat_minimum": str(row.get("수능최저(csat_minimum)", "")).strip(),
                    "schedule": [x.strip() for x in str(row.get("일정(schedule)", "")).split("/") if x.strip()],
                    "notes": str(row.get("주의사항(notes)", "")).strip(),
                    "confidence": str(row.get("confidence", "")).strip() or "manual",
                    "source": {"file": kb.get("universities", {}).get(uni, {}).get(year, {}).get("source_file", "manual")}
                })

            # overwrite routes in KB payload
            payload = kb["universities"][uni][year]
            payload["routes"] = new_routes
            payload["parser"] = {"mode": "manual_override", "route_count": len(new_routes)}
            upsert_university_kb(kb, uni, year, payload)
            save_kb(kb)
            st.success("수정 저장 완료 ✅")

# =========================================================
# TAB 2: Recommendation + Roadmap
# =========================================================
with tab2:
    st.subheader("🧭 전형 추천 + 8주 로드맵")
    kb = load_kb()
    routes = get_university_routes(kb, uni, year)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 1) 내 프로필 입력")
        gpa = st.slider("내신 등급(대략)", 1.0, 9.0, 3.5, 0.1, help="1에 가까울수록 상위권")
        ecs_strength = st.slider("비교과/활동 강도", 1, 5, 3)
        interview = st.slider("면접 자신감", 1, 5, 3)
        nonsul = st.slider("논술 자신감(해당 시)", 1, 5, 2)
        international = st.slider("국제/언어/글로벌 역량", 1, 5, 3)
        reco_ok = st.checkbox("학교 추천(추천형/학교추천 등) 가능", value=False)
        qualification = st.checkbox("기회균형/고른기회 등 자격 전형 해당", value=False)

        profile = {
            "gpa": gpa,
            "ecs_strength": ecs_strength,
            "interview": interview,
            "nonsul": nonsul,
            "international": international,
            "reco_ok": reco_ok,
            "qualification": qualification,
        }

    with right:
        st.markdown("### 2) 수시 세부전형 선택")
        st.caption("‘잘 모르겠어요’를 고르면 KB에서 자동 추천합니다.")
        options = ["잘 모르겠어요(추천받기)"]
        if routes:
            # KB에서 추출된 전형명
            options += sorted(list({r.get("route_name","").strip() for r in routes if r.get("route_name","").strip()}))
        else:
            options += ["학생부종합", "학생부교과", "논술전형", "기회균형", "특기자/실기"]

        chosen = st.selectbox("세부전형", options, index=0)

        st.markdown("### 3) 추천 Top3")
        if not routes:
            st.warning("KB가 없어서 추천이 제한적이야. 먼저 ‘전형 정보(KB) 구축’ 탭에서 PDF 분석→KB 저장을 해줘.")
        else:
            scored = []
            for r in routes:
                s = score_route_by_profile(r, profile)
                scored.append((s, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            top3 = scored[:3]

            for i, (s, r) in enumerate(top3, start=1):
                st.markdown(f"#### #{i} ✅ {r.get('route_name')}")
                st.write(f"- 전형요소: {', '.join(r.get('key_elements', []) or [])}")
                ratio_txt = " | ".join(r.get("evaluation_ratio", []) or [])
                st.write(f"- 반영비율: {ratio_txt if ratio_txt else '요강 확인'}")
                st.write(f"- 수능최저: {r.get('csat_minimum', '요강 확인')}")
                if r.get("notes"):
                    st.info(r["notes"])
                st.caption(f"추천 점수: {s:.2f}")

            # 최종 전형 결정
            st.divider()
            st.markdown("### 4) 최종 전형 선택 → 8주 로드맵 생성")

            if chosen == "잘 모르겠어요(추천받기)":
                final_route = top3[0][1] if top3 else None
                st.success(f"자동 선택: {final_route.get('route_name')}" if final_route else "자동 선택 실패")
            else:
                # 사용자가 특정 전형을 선택했으면 그 전형 우선
                final_route = next((r for r in routes if r.get("route_name") == chosen), None)
                if final_route is None and top3:
                    final_route = top3[0][1]

            if final_route is None:
                st.error("전형을 결정할 수 없어. KB/선택을 확인해줘.")
            else:
                if st.button("📅 8주 로드맵 만들기", type="primary", use_container_width=True):
                    with st.spinner("8주 로드맵 생성 중..."):
                        plan = roadmap_llm_refine(client, model, uni, final_route, profile) if client else roadmap_template(uni, final_route.get("route_name", ""))

                    st.success(f"{uni} · {final_route.get('route_name')} — 8주 로드맵 완료")
                    for w in plan:
                        with st.expander(f"Week {w['week']} — {w['goal']}"):
                            for t in w["tasks"]:
                                st.write(f"- {t}")

                    st.download_button(
                        "⬇️ 로드맵 JSON 다운로드",
                        data=json.dumps(plan, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=f"Y-Compass_{uni}_{year}_{final_route.get('route_name','route')}_8weeks.json",
                        mime="application/json",
                        use_container_width=True
                    )

# =========================================================
# TAB 3: Export / Import / Reset
# =========================================================
with tab3:
    st.subheader("🗂️ DB 관리/내보내기")
    kb = load_kb()
    st.markdown("### 현재 KB 상태")
    uni_keys = list(kb.get("universities", {}).keys())
    st.write({"universities": uni_keys, "updated_at": kb.get("updated_at")})

    st.markdown("### 📤 KB JSON 다운로드")
    st.download_button(
        "⬇️ admission_kb.json 다운로드",
        data=json.dumps(kb, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="admission_kb.json",
        mime="application/json",
        use_container_width=True
    )

    st.markdown("### 📥 KB JSON 업로드(복구/이전)")
    uploaded_kb = st.file_uploader("admission_kb.json 업로드", type=["json"])
    if uploaded_kb is not None:
        try:
            payload = json.loads(uploaded_kb.read().decode("utf-8"))
            if isinstance(payload, dict) and "universities" in payload:
                if st.button("💾 업로드한 KB로 덮어쓰기", use_container_width=True):
                    save_kb(payload)
                    st.success("KB 복구/이전 완료 ✅")
            else:
                st.error("KB JSON 형식이 아니야(필드 universities 필요).")
        except Exception:
            st.error("JSON 파싱 실패. 파일이 깨졌는지 확인해줘.")

    st.markdown("### 🧨 (주의) KB 초기화")
    if st.button("KB 초기화(모든 저장 전형 삭제)", use_container_width=True):
        ensure_data_dir()
        if os.path.exists(KB_PATH):
            os.remove(KB_PATH)
        st.success("KB 초기화 완료. (다시 PDF 업로드→분석하면 됨)")

st.divider()
st.markdown("#### ✅ 이 버전에서 네 요청이 어떻게 ‘반영’됐는지 요약")
st.markdown("""
- **PDF 파서(정규식+LLM 하이브리드)**:  
  - 정규식으로 전형 섹션 감지 → 전형요소/비율/%/최저/일정 키워드 추출 → 표 생성  
  - OpenAI Key 넣으면: LLM이 추출 결과를 **‘공식 전형명/비율 문장/주의사항’ 중심으로 정돈**(환각 금지 프롬프트)
- **전형 DB JSON 저장/즉시 로딩**:  
  - `data/admission_kb.json`에 저장 → 앱 실행 시 `load_kb()`로 즉시 로딩  
  - 배포 환경에서 DB가 날아갈 수 있으니 **KB 다운로드/업로드(복구)**까지 제공
- **“몰라요” UX + 추천/플랜 연결**:  
  - ‘잘 모르겠어요(추천받기)’ 선택 시 KB에서 자동 Top3 추천 → 8주 로드맵 생성
""")
