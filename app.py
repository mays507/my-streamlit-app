import requests
import streamlit as st

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="🎬 나와 어울리는 영화는?", page_icon="🎬", layout="wide")

POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# 선택지(A/B/C/D) → 장르 그룹 매핑
# A: 로맨스/드라마, B: 액션/어드벤처, C: SF/판타지, D: 코미디
GENRE_GROUPS = {
    "romance_drama": {"label": "로맨스/드라마", "with_genres": "18|10749", "ids": [18, 10749]},
    "action": {"label": "액션/어드벤처", "with_genres": "28", "ids": [28]},
    "sf_fantasy": {"label": "SF/판타지", "with_genres": "878|14", "ids": [878, 14]},
    "comedy": {"label": "코미디", "with_genres": "35", "ids": [35]},
}

# =========================
# 사이드바: API Key 입력
# =========================
with st.sidebar:
    st.header("🔑 TMDB API 설정")
    # ✅ 사용자가 준 키를 기본값으로 넣어둠 (실서비스에서는 st.secrets 사용 권장)
    api_key = st.text_input(
        "TMDB API Key",
        value="7e09d0673fe06eca4b69f84a10269574",
        type="password",
        help="TMDB API 키를 입력하세요. (실무에선 코드/깃에 키를 올리지 말고 st.secrets 권장)",
    )
    st.caption("키가 맞는데도 안 되면 네트워크/방화벽 때문에 TMDB 접속이 막힌 경우도 있어요.")

# =========================
# 질문 데이터
# =========================
st.title("🎬 나와 어울리는 영화는?")
st.write("대학생 버전 심리테스트! 5개 질문에 답하면, 결과에 맞춰 TMDB 인기 영화 5편을 추천해줄게요 🍿")

questions = [
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

st.divider()

# =========================
# 유틸 함수
# =========================
def option_to_group(option_text: str) -> str:
    """선택지 문자열 맨 앞 A/B/C/D 기준으로 장르 그룹 반환"""
    if not option_text:
        return "romance_drama"
    first = option_text.strip()[0].upper()
    if first == "A":
        return "romance_drama"
    if first == "B":
        return "action"
    if first == "C":
        return "sf_fantasy"
    if first == "D":
        return "comedy"
    return "romance_drama"


def pick_final_group(groups: list[str]) -> str:
    """가장 많이 나온 그룹 선택(동점이면 우선순위로 결정)"""
    counts = {k: 0 for k in GENRE_GROUPS.keys()}
    for g in groups:
        counts[g] += 1

    max_count = max(counts.values())
    tied = [k for k, v in counts.items() if v == max_count]

    # 동점 우선순위(원하는 대로 조정 가능)
    priority = ["romance_drama", "action", "sf_fantasy", "comedy"]
    tied.sort(key=lambda x: priority.index(x))
    return tied[0]


def build_reason(group_key: str, count: int) -> str:
    """추천 이유(짧게)"""
    label = GENRE_GROUPS[group_key]["label"]
    if group_key == "romance_drama":
        return f"감정선·여운 중심 선택이 많아서 {label} 취향이 가장 강해요. (A 선택 {count}/5)"
    if group_key == "action":
        return f"속도감/미션/승부욕 선택이 많아서 {label}가 찰떡이에요. (B 선택 {count}/5)"
    if group_key == "sf_fantasy":
        return f"세계관·상상력·떡밥 회수 쪽을 좋아해서 {label}로 추천해요. (C 선택 {count}/5)"
    if group_key == "comedy":
        return f"웃김/케미/가벼운 텐션을 선호해서 {label}가 제일 어울려요. (D 선택 {count}/5)"
    return f"{label} 성향이 가장 강하게 나타났어요."


@st.cache_data(show_spinner=False)
def fetch_movies(api_key: str, with_genres: str, n: int = 5):
    """TMDB discover API로 인기 영화 n개 가져오기"""
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": api_key,
        "with_genres": with_genres,         # 예: "28" 또는 "18|10749"
        "language": "ko-KR",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "page": 1,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    return results[:n]


# =========================
# 질문 표시
# =========================
answers = []
for i, item in enumerate(questions):
    answers.append(
        st.radio(item["q"], item["options"], key=f"q{i}")
    )

st.divider()

# =========================
# 결과 보기
# =========================
if st.button("결과 보기"):
    if not api_key.strip():
        st.error("사이드바에 TMDB API Key를 입력해야 결과를 볼 수 있어요.")
        st.stop()

    st.subheader("분석 중...")

    # 1) 답변 분석 → 그룹 집계
    picked_groups = [option_to_group(a) for a in answers]
    counts = {k: 0 for k in GENRE_GROUPS.keys()}
    for g in picked_groups:
        counts[g] += 1

    # 2) 최종 장르(그룹) 결정
    final_group = pick_final_group(picked_groups)
    genre_label = GENRE_GROUPS[final_group]["label"]
    with_genres = GENRE_GROUPS[final_group]["with_genres"]
    reason = build_reason(final_group, counts[final_group])

    st.info(f"당신의 결과: **{genre_label}** 🎥\n\n- 추천 이유: {reason}")

    # 3) TMDB에서 영화 가져오기
    try:
        with st.spinner("TMDB에서 인기 영화 5편 불러오는 중..."):
            movies = fetch_movies(api_key=api_key, with_genres=with_genres, n=5)
    except requests.HTTPError as e:
        st.error("TMDB 요청이 실패했어요. API Key가 유효한지/네트워크가 막히지 않았는지 확인해줘요.")
        st.code(str(e))
        st.stop()
    except requests.RequestException as e:
        st.error("네트워크 문제로 TMDB에 접속하지 못했어요. (방화벽/학교망/프록시 등)")
        st.code(str(e))
        st.stop()
    except Exception as e:
        st.error("알 수 없는 오류가 발생했어요.")
        st.code(str(e))
        st.stop()

    # 4) 영화 표시
    st.subheader("🍿 추천 영화 5편")
    if not movies:
        st.warning("해당 장르에서 영화 결과가 비어 있어요. 다른 장르로 다시 시도해봐요.")
        st.stop()

    for m in movies:
        title = m.get("title") or "제목 정보 없음"
        rating = m.get("vote_average", None)
        overview = m.get("overview") or "줄거리(overview) 정보가 없어요."
        poster_path = m.get("poster_path")

        poster_url = f"{POSTER_BASE}{poster_path}" if poster_path else None

        with st.container(border=True):
            col1, col2 = st.columns([1, 2], gap="large")

            with col1:
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.info("포스터가 없어요 🖼️")

            with col2:
                st.markdown(f"### {title}")
                if rating is not None:
                    st.write(f"⭐ 평점: **{float(rating):.1f} / 10**")
                else:
                    st.write("⭐ 평점: 정보 없음")

                st.markdown("**줄거리**")
                st.write(overview)

                st.markdown("**이 영화를 추천하는 이유**")
                st.write(f"- 당신의 선택 패턴이 **{genre_label}** 쪽으로 기울어 있어요.\n"
                         f"- 그래서 이 장르에서 **인기 작품**을 우선 추천했어요.")

    st.caption("※ TMDB 인기(popularity.desc) 기준 추천이며, ko-KR 데이터가 없는 작품은 줄거리/제목이 일부 비어 보일 수 있어요.")
