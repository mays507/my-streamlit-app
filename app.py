import requests
import streamlit as st

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(page_title="🎬 나와 어울리는 영화는?", page_icon="🎬", layout="wide")
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# A/B/C/D -> 장르 그룹
# - A: 로맨스/드라마
# - B: 액션/어드벤처
# - C: SF/판타지
# - D: 코미디
GENRE_GROUPS = {
    "romance_drama": {"label": "로맨스/드라마", "with_genres": "18|10749"},
    "action": {"label": "액션/어드벤처", "with_genres": "28"},
    "sf_fantasy": {"label": "SF/판타지", "with_genres": "878|14"},
    "comedy": {"label": "코미디", "with_genres": "35"},
}

QUESTIONS = [
    {
        "q": "1) 시험 끝나고 갑자기 하루가 비었다. 너의 “힐링 루틴”은?",
        "options": [
            "A. 카페+산책하면서 감정정리(로맨스/드라마)",
            "B. 즉흥 당일치기/클라이밍/액티비티(액션/어드벤처)",
            "C. 집콕하면서 세계관 빵빵한 콘텐츠 정주행(설정덕후)(SF/판타지)",
            "D. 친구랑 밈 주고받고 예능 보며 깔깔(코미디)",
        ],
    },
    {
        "q": "2) 너가 영화에서 제일 중요한 포인트는?",
        "options": [
            "A. “인물 감정선”이 촘촘해야 몰입됨(로맨스/드라마)",
            "B. 손에 땀 나는 “미션/추격/전투”가 있어야 함(액션/어드벤처)",
            "C. “상상력+세계관+떡밥회수”가 제맛(SF/판타지)",
            "D. 대사/상황이 빵 터지는 “웃김”이 우선(코미디)",
        ],
    },
    {
        "q": "3) 조별과제 발표 10분 전, 너의 멘탈 상태는?",
        "options": [
            "A. ‘나 완전 망하면 어쩌지…’ 감정 폭풍(로맨스/드라마)",
            "B. ‘오히려 좋아’ 전투모드로 해결(액션/어드벤처)",
            "C. ‘이건 시뮬레이션이다’ 뇌내 시나리오 돌림(SF/판타지)",
            "D. ‘ㅋㅋㅋㅋ 살려줘’ 드립으로 버팀(코미디)",
        ],
    },
    {
        "q": "4) 좋아하는 주인공 타입은?",
        "options": [
            "A. 상처 있지만 성장하는 섬세한 주인공(로맨스/드라마)",
            "B. 몸으로 부딪히며 판 뒤집는 히어로(액션/어드벤처)",
            "C. 규칙을 발견하고 세계를 해석하는 천재/이방인(SF/판타지)",
            "D. 찐친 케미로 사건을 망치고(?) 해결하는 인싸/허당(코미디)",
        ],
    },
    {
        "q": "5) 영화 엔딩, 너의 취향은?",
        "options": [
            "A. 여운 남는 현실 엔딩… 눈물 한 방울(로맨스/드라마)",
            "B. 다음 편 기대되는 통쾌한 승리 엔딩(액션/어드벤처)",
            "C. “이게 이렇게 연결된다고?” 소름 반전 엔딩(SF/판타지)",
            "D. 엔딩까지 웃겨서 기분 좋게 나가는 엔딩(코미디)",
        ],
    },
]


# ---------------------------
# 헬퍼 함수
# ---------------------------
def option_to_group(option_text: str) -> str:
    if not option_text:
        return "romance_drama"
    first = option_text.strip()[0].upper()
    return {
        "A": "romance_drama",
        "B": "action",
        "C": "sf_fantasy",
        "D": "comedy",
    }.get(first, "romance_drama")


def pick_final_group(group_list: list[str]) -> tuple[str, dict]:
    counts = {k: 0 for k in GENRE_GROUPS.keys()}
    for g in group_list:
        counts[g] += 1

    max_count = max(counts.values())
    tied = [k for k, v in counts.items() if v == max_count]

    # 동점일 때 우선순위(원하면 바꿔도 됨)
    priority = ["romance_drama", "action", "sf_fantasy", "comedy"]
    tied.sort(key=lambda x: priority.index(x))
    return tied[0], counts


def reason_text(group_key: str, counts: dict) -> str:
    label = GENRE_GROUPS[group_key]["label"]
    picked = counts.get(group_key, 0)

    if group_key == "romance_drama":
        return f"감정선/여운을 중시하는 선택이 많아서 {label}가 가장 잘 맞아요. (A 선택 {picked}/5)"
    if group_key == "action":
        return f"속도감·미션·박진감을 선호해서 {label} 취향이 강해요. (B 선택 {picked}/5)"
    if group_key == "sf_fantasy":
        return f"세계관·상상력·떡밥 회수에 끌려서 {label}가 찰떡이에요. (C 선택 {picked}/5)"
    if group_key == "comedy":
        return f"‘웃김’이 1순위라 {label}가 딱 맞아요. (D 선택 {picked}/5)"
    return f"{label} 성향이 가장 강하게 나타났어요. (선택 {picked}/5)"


@st.cache_data(show_spinner=False)
def fetch_top5_movies(api_key: str, with_genres: str):
    # 요구사항 URL(쿼리) 기반으로 params 사용
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": api_key,
        "with_genres": with_genres,        # 예: "28" 또는 "18|10749"
        "language": "ko-KR",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "page": 1,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return (data.get("results") or [])[:5]


# ---------------------------
# UI
# ---------------------------
st.title("🎬 나와 어울리는 영화는?")
st.write("간단한 5문항 심리테스트로 당신의 영화 취향 무드를 분석하고, TMDB 인기 영화 5편을 추천해줄게요 🍿")

# ✅ 키 입력: 1) 사이드바 입력 2) (선택) st.secrets["TMDB_API_KEY"] 자동 사용
with st.sidebar:
    st.header("🔑 TMDB API Key")
    st.caption("키는 깃허브에 올리면 유출돼서, **사이드바 입력**이나 **st.secrets**로 관리하는 게 안전해요.")
    default_key = st.secrets.get("TMDB_API_KEY", "")
    api_key = st.text_input("API Key", value=default_key, type="password", placeholder="여기에 TMDB API Key 붙여넣기")

st.divider()

answers = []
for i, item in enumerate(QUESTIONS):
    ans = st.radio(item["q"], item["options"], key=f"q{i}")
    answers.append(ans)

st.divider()

if st.button("결과 보기"):
    if not api_key.strip():
        st.error("사이드바에 TMDB API Key를 입력해줘!")
        st.stop()

    st.subheader("분석 중...")

    # 1) 사용자 답변 -> 장르 결정
    groups = [option_to_group(a) for a in answers]
    final_group, counts = pick_final_group(groups)

    label = GENRE_GROUPS[final_group]["label"]
    with_genres = GENRE_GROUPS[final_group]["with_genres"]
    why = reason_text(final_group, counts)

    st.info(f"당신의 무드는 **{label}** 쪽!  \n- 추천 이유: {why}")

    # 2) TMDB API로 인기 영화 5개
    try:
        with st.spinner("TMDB에서 인기 영화 5편 가져오는 중..."):
            movies = fetch_top5_movies(api_key=api_key, with_genres=with_genres)
    except requests.HTTPError as e:
        st.error("TMDB 요청이 실패했어. API Key가 맞는지/권한이 있는지 확인해줘.")
        st.code(str(e))
        st.stop()
    except requests.RequestException as e:
        st.error("네트워크 문제로 TMDB에 접속이 안 돼. (학교/기관망 방화벽, 프록시 등 가능)")
        st.code(str(e))
        st.stop()

    if not movies:
        st.warning("해당 장르에서 영화 결과가 비어 있어. 다른 선택으로 다시 시도해줘.")
        st.stop()

    st.subheader("🍿 추천 영화 5편")

    for m in movies:
        title = m.get("title") or "제목 없음"
        rating = m.get("vote_average")
        overview = m.get("overview") or "줄거리 정보가 없어요."
        poster_path = m.get("poster_path")
        poster_url = f"{POSTER_BASE}{poster_path}" if poster_path else None

        with st.container(border=True):
            c1, c2 = st.columns([1, 2], gap="large")

            with c1:
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.info("포스터 없음 🖼️")

            with c2:
                st.markdown(f"### {title}")
                if rating is not None:
                    st.write(f"⭐ 평점: **{float(rating):.1f} / 10**")
                else:
                    st.write("⭐ 평점: 정보 없음")

                st.markdown("**줄거리**")
                st.write(overview)

                st.markdown("**이 영화를 추천하는 이유**")
                st.write(f"- 당신의 답변이 **{label}** 성향으로 가장 많이 모였어요.\n"
                         f"- 그래서 해당 장르에서 **인기작(조회/화제성 중심)** 위주로 골랐어!")

    st.caption("※ TMDB 인기(popularity.desc) 기준 추천이며, ko-KR 데이터가 없는 작품은 줄거리/제목이 비어 보일 수 있어요.")
