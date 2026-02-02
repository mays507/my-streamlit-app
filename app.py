import streamlit as st
import requests
from collections import Counter
from typing import Dict, List, Optional

# =========================
# Page Config
# =========================
st.set_page_config(page_title="🎬 나와 어울리는 영화는?", page_icon="🎬", layout="wide")

POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# A/B/C/D -> 장르 그룹(사용자 요구 유지)
GENRE_GROUPS = {
    "로맨스/드라마": [10749, 18],
    "액션/어드벤처": [28],
    "SF/판타지": [878, 14],
    "코미디": [35],
}
GROUP_PRIORITY = ["로맨스/드라마", "액션/어드벤처", "SF/판타지", "코미디"]

# =========================
# 질문(5개, 4지선다, 대학생 타겟)
# =========================
QUESTIONS = [
    {
        "q": "1) 시험 끝난 금요일 밤, 너의 플랜은?",
        "options": [
            {"label": "A. 잔잔한 감정선 + 여운 남는 이야기로 힐링할래", "group": "로맨스/드라마"},
            {"label": "B. 몸이 먼저 반응하는 쾌감! 시원한 액션 한 방", "group": "액션/어드벤처"},
            {"label": "C. 현실 탈출… 세계관 미친 상상력에 잠기고 싶어", "group": "SF/판타지"},
            {"label": "D. 뇌 비우고 빵 터지는 웃음으로 스트레스 해제", "group": "코미디"},
        ],
    },
    {
        "q": "2) 팀플에서 네 역할은 보통?",
        "options": [
            {"label": "A. 분위기/감정 케어 담당, 갈등 중재도 내가 함", "group": "로맨스/드라마"},
            {"label": "B. 일단 돌파! 실행 플랜 짜서 밀어붙이는 타입", "group": "액션/어드벤처"},
            {"label": "C. ‘이렇게 하면 어떨까?’ 새로운 아이디어 제시 담당", "group": "SF/판타지"},
            {"label": "D. 분위기 메이커. 웃기면서도 핵심은 챙김", "group": "코미디"},
        ],
    },
    {
        "q": "3) 여행을 간다면 가장 끌리는 코스는?",
        "options": [
            {"label": "A. 감성 카페 + 밤 산책 + 사진… 여운 코스", "group": "로맨스/드라마"},
            {"label": "B. 액티비티/트레킹/스포츠… 몸 쓰는 게 최고", "group": "액션/어드벤처"},
            {"label": "C. 테마파크/전시/체험… ‘세계관’ 있는 장소", "group": "SF/판타지"},
            {"label": "D. 맛집 투어 + 친구들이랑 드립 배틀", "group": "코미디"},
        ],
    },
    {
        "q": "4) OTT에서 썸네일 보고 클릭하는 기준은?",
        "options": [
            {"label": "A. 표정/대사 느낌이 좋은 작품 (감정 몰입이 중요)", "group": "로맨스/드라마"},
            {"label": "B. 스케일/폭발/추격전… 한눈에 ‘세다’ 싶으면 클릭", "group": "액션/어드벤처"},
            {"label": "C. 우주/마법/초능력/괴물… 설정이 신박하면 클릭", "group": "SF/판타지"},
            {"label": "D. 표정만 봐도 웃김. 텐션 가벼우면 클릭", "group": "코미디"},
        ],
    },
    {
        "q": "5) 영화 보고 난 뒤 남는 건 보통?",
        "options": [
            {"label": "A. ‘아…’ 하고 마음이 오래 남는 여운/메시지", "group": "로맨스/드라마"},
            {"label": "B. 심장 뛰는 장면들! 액션 시퀀스가 기억남", "group": "액션/어드벤처"},
            {"label": "C. 설정/세계관 분석… 해석 찾아보는 재미", "group": "SF/판타지"},
            {"label": "D. 명장면/명대사로 친구들이랑 계속 놀림", "group": "코미디"},
        ],
    },
]

# =========================
# Helpers
# =========================
def pick_top_group(scores: Counter) -> str:
    if not scores:
        return GROUP_PRIORITY[0]
    max_score = max(scores.values())
    tied = [g for g, s in scores.items() if s == max_score]
    for g in GROUP_PRIORITY:
        if g in tied:
            return g
    return tied[0]

def safe_text(x: Optional[str]) -> str:
    return x.strip() if isinstance(x, str) and x.strip() else ""

@st.cache_data(ttl=60 * 60, show_spinner=False)
def tmdb_discover_movies(api_key: str, genre_id: int, language: str, page: int = 1) -> Dict:
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": api_key,
        "with_genres": genre_id,
        "language": language,
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "include_video": "false",
        "page": page,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def tmdb_movie_details(api_key: str, movie_id: int, language: str) -> Dict:
    # 공식 문서에서 권장하는 방식: append_to_response로 추가 데이터 한번에
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": api_key, "language": language, "append_to_response": "keywords"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def merge_unique_movies(lists: List[List[Dict]], limit: int = 5) -> List[Dict]:
    seen = set()
    merged = []
    i = 0
    while len(merged) < limit:
        progressed = False
        for lst in lists:
            if i < len(lst):
                m = lst[i]
                mid = m.get("id")
                if mid and mid not in seen:
                    seen.add(mid)
                    merged.append(m)
                    progressed = True
                    if len(merged) >= limit:
                        break
        if not progressed:
            break
        i += 1
    return merged[:limit]

def fetch_recommendations(api_key: str, group: str, language: str, need: int = 5) -> List[Dict]:
    genre_ids = GENRE_GROUPS[group]
    per_genre = []

    for gid in genre_ids:
        results = []
        for p in [1, 2]:
            try:
                data = tmdb_discover_movies(api_key, gid, language, page=p)
                results.extend(data.get("results", []))
            except Exception:
                continue

        cleaned = []
        for m in results:
            if not m.get("poster_path"):
                continue
            if not safe_text(m.get("overview")):
                continue
            cleaned.append(m)

        cleaned.sort(key=lambda x: x.get("popularity", 0), reverse=True)
        per_genre.append(cleaned)

    if len(per_genre) == 1:
        return per_genre[0][:need]
    return merge_unique_movies(per_genre, limit=need)

def build_reason(group: str, movie: Dict, details: Dict) -> str:
    # 짧고 납득되는 추천 이유 생성(키워드 활용)
    rating = movie.get("vote_average", 0)
    kws = []
    kw_obj = details.get("keywords", {})
    if isinstance(kw_obj, dict):
        kws = [k.get("name") for k in kw_obj.get("keywords", []) if k.get("name")]
    kws = kws[:3]

    parts = [f"당신의 결과가 **{group}** 쪽이라 이 장르 인기작을 우선 추천했어요."]
    if rating:
        parts.append(f"평점 **{rating:.1f}**로 반응도 좋아요.")
    if kws:
        parts.append(f"키워드: `{', '.join(kws)}`")
    return " ".join(parts)

# =========================
# UI
# =========================
st.title("🎬 나와 어울리는 영화는?")
st.write("5개 질문에 답하면, 당신의 영화 취향을 분석해서 TMDB 인기 영화 5개를 예쁘게 추천해줄게요 🍿")

with st.sidebar:
    st.header("🔑 TMDB API Key")
    default_key = st.secrets.get("TMDB_API_KEY", "")
    api_key = st.text_input("API Key", value=default_key, type="password", placeholder="TMDB API Key 붙여넣기")
    language = st.selectbox("언어(language)", ["ko-KR", "en-US"], index=0)

st.divider()

answers = []
scores = Counter()

st.subheader("📝 질문")
for i, item in enumerate(QUESTIONS):
    labels = [o["label"] for o in item["options"]]
    choice = st.radio(item["q"], labels, index=None, key=f"q{i}")
    answers.append(choice)

    if choice:
        selected = next(o for o in item["options"] if o["label"] == choice)
        scores[selected["group"]] += 1

st.divider()

if st.button("결과 보기", type="primary"):
    if not api_key.strip():
        st.error("사이드바에 TMDB API Key를 입력해줘!")
        st.stop()

    if any(a is None for a in answers):
        st.warning("5개 질문 모두 답해야 결과를 볼 수 있어 🙂")
        st.stop()

    best_group = pick_top_group(scores)

    # --------------------------
    # 1) 결과 제목 (요구사항)
    # --------------------------
    st.markdown(f"## ✨ 당신에게 딱인 장르는: **{best_group}**!")
    st.caption(f"선택 분포: {dict(scores)}")

    # --------------------------
    # 2) 로딩 spinner (요구사항)
    # --------------------------
    with st.spinner("TMDB에서 영화를 불러오는 중..."):
        movies = fetch_recommendations(api_key.strip(), best_group, language, need=5)

    if not movies:
        st.error("추천 영화를 가져오지 못했어. (네트워크/키/데이터 부족)")
        st.stop()

    # --------------------------
    # 3) 3열 카드 레이아웃 (요구사항)
    # --------------------------
    cols = st.columns(3, gap="large")

    for idx, m in enumerate(movies):
        col = cols[idx % 3]

        movie_id = m.get("id")
        title = m.get("title") or m.get("original_title") or "제목 없음"
        rating = m.get("vote_average")
        overview = safe_text(m.get("overview"))
        poster_path = m.get("poster_path")
        poster_url = f"{POSTER_BASE}{poster_path}" if poster_path else None

        # 상세 정보는 expander 안에서 가져오도록 (속도/UX)
        with col:
            with st.container(border=True):
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.info("포스터 없음 🖼️")

                st.markdown(f"### {title}")
                if rating is not None:
                    st.write(f"⭐ **{float(rating):.1f} / 10**")
                else:
                    st.write("⭐ 평점 정보 없음")

                # --------------------------
                # 4) 카드 클릭 -> 상세(Expander) (요구사항)
                # --------------------------
                with st.expander("상세 정보 보기"):
                    # 상세 로딩도 spinner 처리
                    with st.spinner("상세 정보를 불러오는 중..."):
                        details = {}
                        try:
                            details = tmdb_movie_details(api_key.strip(), movie_id, language)
                        except Exception:
                            details = {}

                    # 줄거리
                    st.markdown("**줄거리**")
                    st.write(overview if overview else "줄거리 정보가 없어요.")

                    # 추천 이유
                    st.markdown("**이 영화를 추천하는 이유**")
                    st.write(build_reason(best_group, m, details))

                    # 추가로 보고 싶으면 키워드도 노출
                    kw_obj = details.get("keywords", {})
                    if isinstance(kw_obj, dict):
                        kws = [k.get("name") for k in kw_obj.get("keywords", []) if k.get("name")]
                        if kws:
                            st.markdown("**키워드**")
                            st.write(", ".join(kws[:10]))

    st.caption("※ TMDB 인기(popularity.desc) 기반 추천이며, ko-KR 데이터가 없으면 줄거리/제목이 일부 비어 보일 수 있어요.")
