# app.py
# -----------------------------
# Y-Compass (2026 수시) : PDF 기반 전형 요약 + 전형 추천 + 8주 로드맵
#
# ✅ requirements.txt 추천
# streamlit
# pypdf
# openai>=1.0.0
#
# (선택) pdfplumber  # pypdf가 텍스트를 잘 못 뽑는 PDF가 있을 때 보완용
# -----------------------------

import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st

# -----------------------------
# Optional: OpenAI (있으면 요약/로드맵이 더 자연어로 좋아짐)
# -----------------------------
def get_openai_client(api_key: str):
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def llm_call(client, model: str, system: str, user: str) -> Optional[str]:
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception:
        return None

# -----------------------------
# PDF text extraction
# -----------------------------
def extract_text_from_pdf(file) -> str:
    # 1) pypdf 우선
    try:
        from pypdf import PdfReader
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        joined = "\n".join(texts).strip()
        if len(joined) > 300:
            return joined
    except Exception:
        pass

    # 2) pdfplumber fallback (설치되어 있으면)
    try:
        import pdfplumber
        texts = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                texts.append(t)
        joined = "\n".join(texts).strip()
        return joined
    except Exception:
        return ""

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# -----------------------------
# Domain model
# -----------------------------
@dataclass
class Track:
    uni: str
    name: str
    short: str
    signals: List[str]               # PDF에 있으면 가산점 주는 키워드
    requires_reco: bool = False      # 학교장추천/추천서/고교추천 등
    needs_interview: bool = False
    needs_essay: bool = False
    needs_nonsul: bool = False
    international_focus: bool = False
    low_gpa_ok: bool = False         # 내신 낮아도 비교과로 승부 가능
    gpa_sensitive: bool = True       # 내신이 중요(교과)
    notes: str = ""

# -----------------------------
# 최소 전형 DB (너가 올린 모집요강 기반 “구조화”의 시작점)
# - 실제 서비스에서는: PDF에서 추출한 정보로 이 DB를 자동 업데이트(=너가 말한 “미리 정리해두기”) 하게 확장하면 됨
# -----------------------------
TRACKS: List[Track] = [
    # 연세대
    Track(
        uni="연세대",
        name="학생부교과[추천형]",
        short="내신 중심(정량) + 고교 추천 필요. 교과가 강하면 제일 직관적인 루트.",
        signals=["추천형", "학생부교과", "정량평가", "교과성적", "학교장"],
        requires_reco=True,
        gpa_sensitive=True,
        notes="(요강상) 활동우수형과 중복지원 불가인 케이스가 있음."
    ),
    Track(
        uni="연세대",
        name="학생부종합[활동우수형]",
        short="비교과/활동/탐구 기반. 서류+면접로 종합평가(내신이 전부는 아님).",
        signals=["활동우수형", "학생부종합", "서류평가", "면접"],
        needs_interview=True,
        low_gpa_ok=True,
        gpa_sensitive=False,
        notes="(요강상) 추천형과 택1 구조(중복지원 제한) 케이스."
    ),
    Track(
        uni="연세대",
        name="학생부종합[국제형]",
        short="국제/언어/해외경험/글로벌 역량 강점. 서류+면접 비중.",
        signals=["국제형", "해외고", "검정고시", "수학기간", "국제"],
        international_focus=True,
        needs_interview=True,
        low_gpa_ok=True,
        gpa_sensitive=False,
    ),
    Track(
        uni="연세대",
        name="학생부종합[기회균형]",
        short="기회균형 자격 해당 시 가장 강력한 전략축. (자격 충족 여부가 핵심)",
        signals=["기회균형", "정원외", "지원자격"],
        low_gpa_ok=True,
        gpa_sensitive=False,
    ),
    Track(
        uni="연세대",
        name="논술전형",
        short="내신이 애매해도 논술 실력으로 뒤집는 루트. 논술 훈련이 핵심.",
        signals=["논술전형", "논술시험"],
        needs_nonsul=True,
        gpa_sensitive=False,
        low_gpa_ok=True,
    ),
    Track(
        uni="연세대",
        name="특기자전형(국제/체육 등)",
        short="특기/실적이 명확한 경우. 증빙과 실적의 ‘객관성’이 관건.",
        signals=["특기자", "국제인재", "체육인재", "실기", "실적"],
        gpa_sensitive=False,
        low_gpa_ok=True,
    ),

    # 고려대
    Track(
        uni="고려대",
        name="학생부교과(학교추천전형)",
        short="교과 90%+서류 10% 구조. 추천 필요 + 수능최저(전형/모집단위별) 고려.",
        signals=["학교추천전형", "학생부(교과)", "90%", "서류 10%", "수능 최저"],
        requires_reco=True,
        gpa_sensitive=True,
        notes="(요강상) 학업우수전형과 복수지원 불가(택1) 케이스."
    ),
    Track(
        uni="고려대",
        name="학생부종합(학업우수전형)",
        short="내신+학업역량 중심 종합. 수능최저가 걸릴 수 있어 계획이 중요.",
        signals=["학업우수전형", "학생부종합", "지원자격", "수능 최저"],
        needs_interview=False,   # 모집단위별로 상이할 수 있어 ‘기본값’은 False
        gpa_sensitive=True,
        notes="(요강상) 학교추천전형과 택1 구조(복수지원 불가) 케이스."
    ),
    Track(
        uni="고려대",
        name="학생부종합(계열적합전형)",
        short="전공/계열 적합성(탐구·활동 스토리)로 승부. 비교과 설계가 핵심.",
        signals=["계열적합전형", "학생부종합", "계열"],
        needs_interview=False,
        low_gpa_ok=True,
        gpa_sensitive=False,
    ),
    Track(
        uni="고려대",
        name="학생부종합(고른기회/다문화/재직자/사이버국방 등)",
        short="지원자격 해당 시 강력. 자격요건 충족 여부를 먼저 체크.",
        signals=["고른기회", "다문화", "재직자", "사이버국방"],
        low_gpa_ok=True,
        gpa_sensitive=False,
    ),
    Track(
        uni="고려대",
        name="논술전형",
        short="논술로 승부. 수능최저/유형을 함께 관리해야 실전에서 안전.",
        signals=["논술전형", "논술"],
        needs_nonsul=True,
        gpa_sensitive=False,
        low_gpa_ok=True,
    ),
    Track(
        uni="고려대",
        name="실기/실적(특기자전형)",
        short="실적이 ‘증빙 가능한 강점’일 때 유효. 포트폴리오/증빙 정리 필수.",
        signals=["특기자전형", "실기", "실적", "증빙"],
        gpa_sensitive=False,
        low_gpa_ok=True,
    ),

    # 서울대 (업로드 PDF에서 확인되는 범위 기반 최소 구조)
    Track(
        uni="서울대",
        name="지역균형전형",
        short="학교 추천/균형 선발 축. 수능 관련 기준(응시영역/최저)을 반드시 체크.",
        signals=["지역균형", "수능", "최저학력기준", "응시영역"],
        requires_reco=True,
        gpa_sensitive=True,
    ),
    Track(
        uni="서울대",
        name="일반전형",
        short="종합 역량 기반 선발 축. 모집단위별 전형요소/수능기준 체크가 핵심.",
        signals=["일반전형", "수능", "응시영역", "최저학력기준"],
        needs_interview=False,
        low_gpa_ok=True,
        gpa_sensitive=False,
    ),
]

UNIS = ["서울대", "연세대", "고려대"]

# -----------------------------
# Heuristic scoring (추천 로직)
# -----------------------------
def score_track(track: Track, profile: Dict) -> float:
    gpa = profile["gpa"]                 # 1.0~9.0 (작을수록 좋음)
    ecs = profile["ecs_strength"]        # 1~5
    interview = profile["interview"]     # 1~5
    essay = profile["essay"]             # 1~5
    nonsul = profile["nonsul"]           # 1~5
    intl = profile["international"]      # 1~5
    reco_ok = profile["reco_ok"]         # bool
    qualification = profile["qualification"]  # 기회균형 등 해당 여부(선택)

    s = 0.0

    # 추천 가능 여부
    if track.requires_reco and not reco_ok:
        s -= 2.5
    if track.requires_reco and reco_ok:
        s += 0.8

    # 내신 적합
    if track.gpa_sensitive:
        # gpa 낮을수록(=1~2등급) 유리
        if gpa <= 2.5:
            s += 2.2
        elif gpa <= 4.0:
            s += 1.2
        elif gpa <= 6.0:
            s += 0.2
        else:
            s -= 1.2
    else:
        # 내신이 완벽하지 않아도 되는 축
        if gpa >= 5.0 and track.low_gpa_ok:
            s += 0.8

    # 비교과/활동
    if "종합" in track.name or "계열" in track.name or "활동" in track.name:
        s += (ecs - 3) * 0.8

    # 면접/에세이/논술
    if track.needs_interview:
        s += (interview - 3) * 0.7
    if track.needs_essay:
        s += (essay - 3) * 0.6
    if track.needs_nonsul:
        s += (nonsul - 3) * 1.0

    # 국제
    if track.international_focus:
        s += (intl - 3) * 0.9

    # 기회균형/특수자격 가점(사용자가 해당하면)
    if qualification and ("기회" in track.name or "고른기회" in track.name or "다문화" in track.name or "재직자" in track.name):
        s += 2.0

    return s

# -----------------------------
# 8-week roadmap generator (템플릿 + LLM 강화)
# -----------------------------
def roadmap_template(track_name: str, uni: str) -> List[Dict]:
    # 전형 타입에 따라 강조 포인트가 달라지도록
    is_nonsul = "논술" in track_name
    is_gw = ("기회" in track_name) or ("고른기회" in track_name)
    is_gw = bool(is_gw)
    is_gyogwa = ("교과" in track_name) or ("학교추천" in track_name) or ("추천형" in track_name)
    is_jonghap = ("종합" in track_name) or ("활동" in track_name) or ("계열" in track_name) or ("일반전형" in track_name)

    weeks = []
    for w in range(1, 9):
        item = {"week": w, "goal": "", "tasks": []}

        if w == 1:
            item["goal"] = "전형 구조/자격/제출물 체크 + 나의 스펙 진단"
            item["tasks"] = [
                f"{uni} {track_name} 모집요강에서 자격/제출/일정 체크리스트 만들기",
                "내신/비교과/수능/포트폴리오 현황을 한 장으로 정리(갭 분석)",
                "지원 학과 3~5개 후보 확정 + 리스크(수능최저/추천/면접) 표시"
            ]

        elif w == 2:
            item["goal"] = "스토리라인(왜 이 전공/왜 이 학교) 초안 만들기"
            item["tasks"] = [
                "학생부/활동/수상/독서/탐구를 ‘전공 적합성’ 관점으로 재배열",
                "핵심 키워드 5개(관심 분야/문제의식/활동근거/성과/성장) 뽑기",
                "자기소개서/활동기록표/면접 대비용 ‘1분 자기소개’ 초안"
            ]

        elif w == 3:
            if is_nonsul:
                item["goal"] = "논술 베이스 구축: 유형 파악 + 기출 1회독"
                item["tasks"] = [
                    "해당 대학/계열 논술 기출 유형 분석(주제/채점포인트/분량)",
                    "기출 2~3개 답안 작성 → 시간 관리 기준 만들기",
                    "약점(개요/근거/문장력/계산)을 체크리스트로 고정"
                ]
            else:
                item["goal"] = "서류 품질 올리기: 근거/정합성/디테일 강화"
                item["tasks"] = [
                    "활동 3개를 ‘문제-행동-결과-배움’ 구조로 리라이팅",
                    "전공 적합성 근거(탐구/프로젝트/읽은 자료) 3개 확정",
                    "면접 예상질문 20개 뽑고 답변 포맷 만들기(결론-근거-예시)"
                ]

        elif w == 4:
            if is_gyogwa:
                item["goal"] = "교과 전략 정교화 + 수능/내신 리스크 관리"
                item["tasks"] = [
                    "교과 성적 산출/반영 과목 확인 후 강점 과목 표 만들기",
                    "수능최저가 있으면 4주 단기 플랜(영역별 목표 등급) 설정",
                    "추천 필요 시 담임/진학부 협의 일정 잡기(추천 가능성 확정)"
                ]
            else:
                item["goal"] = "서류/면접(또는 논술) 실전 난이도로 끌어올리기"
                item["tasks"] = [
                    "모의면접 1회(녹화) → 말버릇/논리/구체성 피드백",
                    "포트폴리오/증빙 정리(파일명 규칙, 한 폴더에 모으기)",
                    "지원 학과별 ‘왜 여기?’ 맞춤 문장 3개씩 만들기"
                ]

        elif w == 5:
            item["goal"] = "전형별 ‘결정’ 단계: 지원 조합(플랜A/B) 확정"
            item["tasks"] = [
                "수시 6장 전략: 상향/적정/안정 밸런스 재점검",
                "각 전형별 ‘합격 포인트’ 3줄로 요약(내가 이길 수 있는 이유)",
                "부족한 부분 1개를 정하고 2주 동안 집중 보완(예: 면접/논술/수능)"
            ]

        elif w == 6:
            if is_nonsul:
                item["goal"] = "논술 실전 주간: 기출+실전 모의 3회"
                item["tasks"] = [
                    "실전 시간(시험과 동일)으로 기출 2회 + 모의 1회",
                    "답안 첨삭 포인트: 논지 일관성/근거 질/문장 명료성",
                    "자주 틀리는 패턴 5개를 ‘금지 규칙’으로 만들기"
                ]
            else:
                item["goal"] = "면접/서류 최종 압축: 말이 ‘증거’가 되게"
                item["tasks"] = [
                    "빈출 질문(지원동기/전공적합/협업/갈등/성장) 최종 스크립트",
                    "내 활동의 숫자/성과/역할을 ‘객관적 표현’으로 정리",
                    "모의면접 2회 + 예상 꼬리질문 리스트 업데이트"
                ]

        elif w == 7:
            item["goal"] = "원서/서류 제출 전 체크리스트 완료"
            item["tasks"] = [
                "제출 파일/양식/글자수/개인정보 노출 여부 최종 점검",
                "추천/학교 제출(해당 시) 마감일 역산해서 완료",
                "실수 방지: 파일명 규칙 통일 + 최종본 백업(클라우드/USB)"
            ]

        elif w == 8:
            item["goal"] = "마무리: 시험/면접/서류 ‘실전 컨디션’ 세팅"
            item["tasks"] = [
                "실전 루틴 만들기(수면/식사/이동/준비물) + 체크리스트 출력",
                "지원 대학별 마지막 10분 요약 노트(전공/활동/질문) 만들기",
                "결과 대기 플랜(추가합격/정시 전환 대비)까지 설계"
            ]

        if is_gw:
            item["tasks"].append("※ (자격 전형) 자격 증빙/서류 누락이 ‘즉탈’ 포인트라서, 증빙 체크를 가장 먼저 고정")

        weeks.append(item)
    return weeks

def roadmap_with_llm(client, model: str, uni: str, track: str, user_profile: Dict, pdf_summary: str) -> List[Dict]:
    base = roadmap_template(track, uni)
    if client is None:
        return base

    system = "너는 한국 입시 컨설턴트이자 학습 코치다. 사용자의 전형에 맞춘 8주 로드맵을 매우 구체적으로 작성한다."
    user = f"""
대학: {uni}
전형: {track}

사용자 프로필:
- 내신(등급): {user_profile['gpa']}
- 비교과 강도(1~5): {user_profile['ecs_strength']}
- 면접 자신감(1~5): {user_profile['interview']}
- 글쓰기(1~5): {user_profile['essay']}
- 논술(1~5): {user_profile['nonsul']}
- 국제역량(1~5): {user_profile['international']}
- 추천 가능 여부: {user_profile['reco_ok']}
- (기회/자격 전형 해당): {user_profile['qualification']}

모집요강 요약(발췌/정리):
{pdf_summary[:2000]}

요청:
- 8주 로드맵을 주차별로 goal 1개 + tasks 3~6개로 작성
- tasks는 “산출물(결과물)” 형태로 쓰기 (예: 체크리스트, 스크립트, 기출 n회, 포트폴리오 폴더)
- JSON 배열로만 출력: [{{"week":1,"goal":"...","tasks":["..."]}}, ...]
"""
    out = llm_call(client, model, system, user)
    if not out:
        return base

    try:
        # LLM이 JSON만 주면 파싱
        data = json.loads(out)
        if isinstance(data, list) and len(data) == 8:
            return data
        return base
    except Exception:
        return base

# -----------------------------
# PDF 요약/전형정보 추출 (간단 버전)
# -----------------------------
def quick_pdf_summary(text: str, client=None, model="gpt-4o-mini") -> str:
    if not text:
        return "PDF에서 텍스트를 충분히 추출하지 못했어요. (스캔본일 수 있음) 다른 PDF로 시도하거나 pdfplumber를 추가해보세요."

    excerpt = text[:6000]
    if client is None:
        # 키워드 기반 미니 요약(LLM 없을 때)
        keys = ["전형", "모집", "일정", "수능", "최저", "면접", "논술", "서류", "추천", "자격"]
        hit = []
        for k in keys:
            if k in excerpt:
                hit.append(k)
        return f"텍스트 추출 OK. (키워드 감지: {', '.join(hit)})\n\n요약은 OpenAI API Key를 넣으면 더 정확하게 자동 생성됩니다."

    system = "너는 대학 입학전형 요강을 읽고 핵심만 뽑아주는 분석가다."
    user = f"""
아래는 '수시 모집요강' 일부 텍스트다. 다음 형식으로 아주 간결하게 정리해줘.

형식:
- 전형 큰 분류(교과/종합/논술/특기자/기회) 별로: (1) 핵심 전형요소 (2) 수능최저 유무 (3) 주의사항(중복지원/추천/자격)
- 일정(원서/시험/발표)에서 눈에 띄는 포인트 3개만

텍스트:
{excerpt}
"""
    out = llm_call(client, model, system, user)
    return out or "요약 생성에 실패했어요. (API Key/모델/요금 한도 확인)"

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Y-Compass (2026 수시) - PDF 기반 전형 추천", layout="wide")

st.title("🎓 Y-Compass (2026 수시) — PDF 기반 전형 추천 + 8주 로드맵")
st.caption("서울대/연세대/고려대 모집요강 PDF를 올리면 전형 정보를 정리하고, 너에게 맞는 전형과 8주 플랜을 제안합니다.")

with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("OpenAI API Key (선택)", type="password", help="넣으면 PDF 요약/로드맵이 훨씬 자연어로 정확해져요.")
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    st.divider()
    st.subheader("📌 지원 대학 선택")
    uni = st.selectbox("대학", UNIS, index=1)

client = get_openai_client(api_key) if api_key else None

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("1) 모집요강 PDF 업로드")
    uploaded = st.file_uploader(
        "서울대/연세대/고려대 2026 수시 모집요강 PDF 업로드",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        st.success(f"{len(uploaded)}개 파일 업로드 완료")
    else:
        st.info("PDF를 올리면, 해당 PDF에서 텍스트를 추출해 전형 정보를 요약합니다.")

    st.subheader("2) 내 프로필 입력 (추천 정확도 업)")
    gpa = st.slider("내신 등급(대략)", min_value=1.0, max_value=9.0, value=3.5, step=0.1, help="1에 가까울수록 상위권")
    ecs_strength = st.slider("비교과/활동 강도", 1, 5, 3, help="탐구/동아리/프로젝트/수상/리더십/봉사 등 종합")
    interview = st.slider("면접 자신감", 1, 5, 3)
    essay = st.slider("글쓰기/서류 작성 자신감", 1, 5, 3)
    nonsul = st.slider("논술 자신감(해당 시)", 1, 5, 2)
    international = st.slider("국제/언어/글로벌 역량", 1, 5, 3)
    reco_ok = st.checkbox("학교 추천(추천형/학교추천 등) 가능", value=False)
    qualification = st.checkbox("기회균형/고른기회/특수 자격 전형 해당", value=False)

    profile = {
        "gpa": gpa,
        "ecs_strength": ecs_strength,
        "interview": interview,
        "essay": essay,
        "nonsul": nonsul,
        "international": international,
        "reco_ok": reco_ok,
        "qualification": qualification,
    }

with colB:
    st.subheader("3) PDF 분석 & 전형 요약")
    pdf_texts = {}
    pdf_summaries = {}

    if uploaded:
        for f in uploaded:
            # 파일 포인터는 한 번 읽으면 끝나서, 추출 전에 seek(0)
            try:
                f.seek(0)
            except Exception:
                pass
            raw = extract_text_from_pdf(f)
            pdf_texts[f.name] = raw
            pdf_summaries[f.name] = quick_pdf_summary(raw, client=client, model=model)

        pick = st.selectbox("요약 볼 파일", list(pdf_summaries.keys()))
        st.text_area("요약 결과", pdf_summaries[pick], height=260)

        with st.expander("원문 텍스트(일부) 보기"):
            st.write(clean_text(pdf_texts[pick])[:2000] + " ...")
    else:
        st.warning("아직 PDF가 없어서 요약을 못 만들었어. PDF 업로드하면 여기에 뜰 거야.")

    st.subheader("4) 전형 추천 (Top 3)")
    candidate_tracks = [t for t in TRACKS if t.uni == uni]

    # PDF 키워드 매칭으로 가산점(업로드된 PDF에 해당 대학 키워드가 있으면)
    joined_pdf_text = ""
    if uploaded:
        joined_pdf_text = "\n".join(pdf_texts.values())

    scored = []
    for t in candidate_tracks:
        base = score_track(t, profile)
        # PDF 키워드가 텍스트에 있으면 +0.1씩
        if joined_pdf_text:
            hits = sum(1 for k in t.signals if k in joined_pdf_text)
            base += min(1.2, hits * 0.12)
        scored.append((base, t))

    scored.sort(key=lambda x: x[0], reverse=True)
    top3 = scored[:3]

    for rank, (s, t) in enumerate(top3, start=1):
        st.markdown(f"### #{rank} ✅ {t.name}")
        st.write(t.short)
        if t.notes:
            st.info(t.notes)
        st.caption(f"추천 점수: {s:.2f}")

    st.divider()

    st.subheader("5) 최종 선택 전형 → 8주 로드맵 생성")
    chosen_name = st.selectbox(
        "전형 선택(추천 Top3 중 하나를 고르거나 직접 선택)",
        [t.name for _, t in top3] + ["(직접 선택)"],
        index=0
    )

    if chosen_name == "(직접 선택)":
        chosen_name = st.selectbox("직접 선택", [t.name for t in candidate_tracks])

    chosen_track = next((t for t in candidate_tracks if t.name == chosen_name), None)

    # LLM에 줄 pdf 요약은 "가장 관련 있어 보이는" 파일 요약을 하나 선택
    pdf_summary_for_llm = ""
    if uploaded:
        # 대학 이름 포함된 파일이 있으면 우선
        uni_hint = {"서울대": "서울", "연세대": "연세", "고려대": "고려"}.get(uni, "")
        matched = [name for name in pdf_summaries.keys() if uni_hint and uni_hint in name]
        if matched:
            pdf_summary_for_llm = pdf_summaries[matched[0]]
        else:
            pdf_summary_for_llm = list(pdf_summaries.values())[0]

    if st.button("📅 8주 로드맵 만들기", use_container_width=True):
        if not chosen_track:
            st.error("전형 선택이 이상해. 다시 선택해줘.")
        else:
            plan = roadmap_with_llm(
                client=client,
                model=model,
                uni=uni,
                track=chosen_track.name,
                user_profile=profile,
                pdf_summary=pdf_summary_for_llm
            )

            st.success(f"{uni} · {chosen_track.name} — 8주 로드맵 생성 완료")
            for w in plan:
                with st.expander(f"Week {w['week']} — {w['goal']}"):
                    for task in w["tasks"]:
                        st.write(f"- {task}")

            # JSON 다운로드
            st.download_button(
                "⬇️ 로드맵 JSON 다운로드",
                data=json.dumps(plan, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"Y-Compass_{uni}_{chosen_track.name}_8weeks.json",
                mime="application/json",
                use_container_width=True
            )

st.divider()
st.subheader("🧠 너가 말한 ‘확장 아이디어’가 이 앱에서 어떻게 구현되는지 (설계 메모)")

st.markdown("""
- **(지금)** PDF 업로드 → 텍스트 추출 → 요약/전형 추천/8주 플랜  
- **(다음 단계)** 학교별로 PDF를 “미리 정리(구조화)”해두기  
  - 방법 A) 입학처/입시 사이트의 **공식 API**가 있으면 그걸 우선 사용(최고)
  - 방법 B) 없다면: 모집요강 PDF를 정규식/LLM으로 파싱해서 **전형 DB(Track JSON)**로 저장  
  - 방법 C) 한정된 학교(서울/연고)로 시작 → 작동 검증 → 학교 수 확장  
- **(핵심 포인트)** 전형 추천은 결국 “구조화된 전형 DB + 사용자 프로필(내신/비교과/면접/논술/추천여부/자격)” 매칭 문제
""")
