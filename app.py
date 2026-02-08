 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app.py b/app.py
index c3f5bdfff134337ce2cafd72ef39be113100696b..4d2a71b9292a1d4f2ff3ce420e4bcaf001b1eac2 100644
--- a/app.py
+++ b/app.py
@@ -1,430 +1,211 @@
-import json
-from collections import Counter
-from typing import Dict, List, Optional, Tuple
-
-import requests
-import streamlit as st
-
-# =========================
-# Page Config
-# =========================
-st.set_page_config(page_title="🎬 나와 어울리는 영화는?", page_icon="🎬", layout="wide")
-
-POSTER_BASE = "https://image.tmdb.org/t/p/w500"
-
-# -------------------------
-# Quiz: 4가지 성향(각 선택지는 장르 선호를 반영)
-# -------------------------
-GENRE_GROUPS = {
-    "로맨스/드라마": [10749, 18],  # Romance + Drama
-    "액션/어드벤처": [28],         # Action
-    "SF/판타지": [878, 14],        # Sci-Fi + Fantasy
-    "코미디": [35],                # Comedy
-}
-GROUP_PRIORITY = ["로맨스/드라마", "액션/어드벤처", "SF/판타지", "코미디"]
-
-QUESTIONS = [
-    {
-        "q": "1) 시험 끝나고 갑자기 하루가 비었다. 너의 ‘힐링 루틴’은?",
-        "options": [
-            {"label": "A. 카페+산책하면서 감정정리(로맨스/드라마)", "group": "로맨스/드라마"},
-            {"label": "B. 즉흥 당일치기/액티비티로 뇌 리셋(액션/어드벤처)", "group": "액션/어드벤처"},
-            {"label": "C. 세계관 빵빵한 작품 정주행(설정덕후)(SF/판타지)", "group": "SF/판타지"},
-            {"label": "D. 친구랑 밈/예능 보며 깔깔(코미디)", "group": "코미디"},
-        ],
-    },
-    {
-        "q": "2) 너가 영화에서 제일 중요한 포인트는?",
-        "options": [
-            {"label": "A. 인물 감정선/여운(로맨스/드라마)", "group": "로맨스/드라마"},
-            {"label": "B. 속도감/미션/추격/전투(액션/어드벤처)", "group": "액션/어드벤처"},
-            {"label": "C. 세계관/떡밥회수/상상력(SF/판타지)", "group": "SF/판타지"},
-            {"label": "D. 대사/상황이 빵 터지는 웃김(코미디)", "group": "코미디"},
-        ],
-    },
-    {
-        "q": "3) 조별과제 발표 10분 전, 너의 멘탈은?",
-        "options": [
-            {"label": "A. 감정 폭풍… 내적 드라마 시작(로맨스/드라마)", "group": "로맨스/드라마"},
-            {"label": "B. 전투모드 ON, 해결부터(액션/어드벤처)", "group": "액션/어드벤처"},
-            {"label": "C. 뇌내 시뮬레이션으로 플랜 재구성(SF/판타지)", "group": "SF/판타지"},
-            {"label": "D. 드립으로 버티며 웃음으로 환기(코미디)", "group": "코미디"},
-        ],
-    },
-    {
-        "q": "4) 좋아하는 주인공 타입은?",
-        "options": [
-            {"label": "A. 상처 있지만 성장하는 섬세한 주인공(로맨스/드라마)", "group": "로맨스/드라마"},
-            {"label": "B. 몸으로 판 뒤집는 히어로(액션/어드벤처)", "group": "액션/어드벤처"},
-            {"label": "C. 규칙을 발견하고 세계를 해석하는 이방인(SF/판타지)", "group": "SF/판타지"},
-            {"label": "D. 케미로 사건을 망치고(?) 해결하는 허당/인싸(코미디)", "group": "코미디"},
-        ],
-    },
-    {
-        "q": "5) 영화 엔딩 취향은?",
-        "options": [
-            {"label": "A. 여운 남는 현실 엔딩(로맨스/드라마)", "group": "로맨스/드라마"},
-            {"label": "B. 통쾌한 승리/클리프행어 엔딩(액션/어드벤처)", "group": "액션/어드벤처"},
-            {"label": "C. 소름 반전/떡밥 회수 엔딩(SF/판타지)", "group": "SF/판타지"},
-            {"label": "D. 끝까지 웃기고 기분 좋은 엔딩(코미디)", "group": "코미디"},
-        ],
-    },
-]
-
-# =========================
-# Helpers
-# =========================
-def safe_text(x: Optional[str]) -> str:
-    return x.strip() if isinstance(x, str) and x.strip() else ""
-
-
-def pick_top_group(scores: Counter) -> str:
-    if not scores:
-        return GROUP_PRIORITY[0]
-    max_score = max(scores.values())
-    tied = [g for g, s in scores.items() if s == max_score]
-    for g in GROUP_PRIORITY:
-        if g in tied:
-            return g
-    return tied[0]
-
-
-@st.cache_data(ttl=60 * 60, show_spinner=False)
-def tmdb_discover_movies(api_key: str, with_genres: str, language: str, page: int = 1) -> Dict:
-    url = "https://api.themoviedb.org/3/discover/movie"
-    params = {
-        "api_key": api_key,
-        "with_genres": with_genres,       # 예: "28" 또는 "18|10749"
-        "language": language,
-        "sort_by": "popularity.desc",
-        "include_adult": "false",
-        "include_video": "false",
-        "page": page,
-    }
-    r = requests.get(url, params=params, timeout=15)
-    r.raise_for_status()
-    return r.json()
-
-
-@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
-def tmdb_movie_details(api_key: str, movie_id: int, language: str) -> Dict:
-    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
-    params = {"api_key": api_key, "language": language, "append_to_response": "keywords"}
-    r = requests.get(url, params=params, timeout=15)
-    r.raise_for_status()
-    return r.json()
-
-
-def fetch_recommendations(api_key: str, group: str, language: str, n: int = 5) -> List[Dict]:
-    # 그룹이 복수 장르(로맨스/드라마, SF/판타지)면 섞어서 다양성 확보
-    ids = GENRE_GROUPS[group]
-    with_genres_list = ["|".join(map(str, ids))] if len(ids) > 1 else [str(ids[0])]
-
-    # discover 결과를 충분히 가져와서 포스터/줄거리 있는 애만 추리기
-    movies: List[Dict] = []
-    for with_genres in with_genres_list:
-        for p in [1, 2]:
-            data = tmdb_discover_movies(api_key, with_genres, language, page=p)
-            movies.extend(data.get("results", []) or [])
-
-    cleaned = []
-    seen = set()
-    for m in movies:
-        mid = m.get("id")
-        if not mid or mid in seen:
-            continue
-        if not m.get("poster_path"):
-            continue
-        if not safe_text(m.get("overview")):
-            continue
-        seen.add(mid)
-        cleaned.append(m)
-
-    cleaned.sort(key=lambda x: x.get("popularity", 0), reverse=True)
-    return cleaned[:n]
-
-
-def openai_final_pick(
-    openai_api_key: str,
-    user_answers: List[str],
-    best_group: str,
-    group_scores: Dict[str, int],
-    movies: List[Dict],
-    model: str,
-) -> Dict:
-    """
-    OpenAI Responses API로 '최종 1개'를 고르는 함수.
-    - 영화 후보 5개 중 1개만 선택
-    - JSON만 반환하도록 강제하고 파싱
-    """
-    # 후보 영화 요약(LLM 입력용, 너무 길어지지 않게)
-    movie_summaries = []
-    for m in movies:
-        movie_summaries.append(
-            {
-                "id": m.get("id"),
-                "title": m.get("title") or m.get("original_title"),
-                "vote_average": m.get("vote_average"),
-                "release_date": m.get("release_date"),
-                "overview": (m.get("overview") or "")[:600],
-            }
-        )
-
-    prompt = f"""
-너는 '대학생 대상 영화 취향 심리테스트'의 최종 추천 전문가야.
-사용자의 답변과 성향(장르)을 바탕으로, 아래 '후보 영화 5개' 중에서
-사용자가 "진짜로 좋아할 확률"이 가장 높은 영화 단 1개를 골라줘.
-
-규칙:
-- 반드시 후보 5개 중 1개만 선택
-- 아래 JSON 스키마로만 출력 (다른 문장/설명 금지)
-- reason은 2~4문장, 한국어, 구체적으로(답변 패턴과 영화 특징을 연결)
-
-JSON 스키마:
-{{
-  "movie_id": number,
-  "title": string,
-  "reason": string,
-  "confidence": number
-}}
-
-사용자 답변(5개):
-{json.dumps(user_answers, ensure_ascii=False)}
-
-심리테스트 결과 장르:
-{best_group}
-
-장르 점수:
-{json.dumps(group_scores, ensure_ascii=False)}
-
-후보 영화 5개:
-{json.dumps(movie_summaries, ensure_ascii=False)}
-""".strip()
-
-    url = "https://api.openai.com/v1/responses"
-    headers = {
-        "Authorization": f"Bearer {openai_api_key}",
-        "Content-Type": "application/json",
-    }
-
-    payload = {
-        "model": model,
-        "input": [
-            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
-        ],
-        # JSON만 주도록 강제(모델이 지원하는 경우 잘 지켜짐)
-        "text": {"format": {"type": "json_object"}},
-    }
-
-    r = requests.post(url, headers=headers, json=payload, timeout=30)
-    r.raise_for_status()
-    data = r.json()
-
-    # Responses API output 텍스트 추출
-    # 일반적으로 data["output"][...]["content"][...]["text"] 형태
-    text_out = ""
-    for out_item in data.get("output", []):
-        for c in out_item.get("content", []):
-            if c.get("type") in ("output_text", "text") and c.get("text"):
-                text_out += c["text"]
-
-    text_out = text_out.strip()
-    if not text_out:
-        raise ValueError("OpenAI 응답 텍스트가 비어있어요.")
-
-    # JSON 파싱
-    try:
-        result = json.loads(text_out)
-    except json.JSONDecodeError:
-        # 혹시 모델이 주변 텍스트를 섞었으면 마지막 JSON 블록만 시도
-        start = text_out.find("{")
-        end = text_out.rfind("}")
-        if start != -1 and end != -1 and end > start:
-            result = json.loads(text_out[start : end + 1])
-        else:
-            raise
-
-    # 최소 검증
-    if "movie_id" not in result:
-        raise ValueError("OpenAI 결과에 movie_id가 없어요.")
-    return result
-
-
-# =========================
-# UI
-# =========================
-st.title("🎬 나와 어울리는 영화는?")
-st.write("5문항 심리테스트 → 장르 분석 → TMDB 인기 영화 5편 추천 → (추가) OpenAI가 최종 1편 ‘진짜 취향픽’까지 골라줘요 🍿")
-
-with st.sidebar:
-    st.header("🔑 API 키 입력")
-    tmdb_key_default = st.secrets.get("TMDB_API_KEY", "")
-    openai_key_default = st.secrets.get("OPENAI_API_KEY", "")
-
-    tmdb_api_key = st.text_input("TMDB API Key", value=tmdb_key_default, type="password", placeholder="TMDB API Key")
-    openai_api_key = st.text_input("OpenAI API Key", value=openai_key_default, type="password", placeholder="OpenAI API Key")
-
-    st.divider()
-    st.subheader("⚙️ 옵션")
-    language = st.selectbox("TMDB 언어", ["ko-KR", "en-US"], index=0)
-
-    # 모델은 환경/계정마다 다를 수 있어서 사용자가 바꿀 수 있게
-    openai_model = st.text_input("OpenAI 모델", value="gpt-4.1-mini", help="계정에서 사용 가능한 모델로 바꿔도 돼요.")
-
-
-st.divider()
-
-# 질문 영역
-answers: List[Optional[str]] = []
-scores = Counter()
-
-st.subheader("📝 질문")
-for i, item in enumerate(QUESTIONS):
-    labels = [o["label"] for o in item["options"]]
-    choice = st.radio(item["q"], labels, index=None, key=f"q{i}")
-    answers.append(choice)
-
-    if choice:
-        selected = next(o for o in item["options"] if o["label"] == choice)
-        scores[selected["group"]] += 1
-
-st.divider()
-
-# 세션 상태(재실행 시 카드 열어도 API를 불필요하게 다시 안 치게)
-if "result_ready" not in st.session_state:
-    st.session_state.result_ready = False
-    st.session_state.best_group = None
-    st.session_state.movies = []
-    st.session_state.llm_pick = None
-
-if st.button("결과 보기", type="primary"):
-    if any(a is None for a in answers):
-        st.warning("5개 질문 모두 선택해줘야 결과를 낼 수 있어 🙂")
-        st.stop()
-    if not tmdb_api_key.strip():
-        st.error("사이드바에 TMDB API Key를 입력해줘!")
-        st.stop()
-
-    best_group = pick_top_group(scores)
-    group_scores = {k: int(v) for k, v in scores.items()}
-
-    with st.spinner("TMDB에서 추천 영화를 불러오는 중..."):
-        movies = fetch_recommendations(tmdb_api_key.strip(), best_group, language, n=5)
-
-    if not movies:
-        st.error("영화 데이터를 가져오지 못했어. (TMDB 키/네트워크/필터 조건 확인)")
-        st.stop()
-
-    llm_pick = None
-    if openai_api_key.strip():
-        try:
-            with st.spinner("OpenAI가 ‘진짜 취향픽’ 1편을 고르는 중..."):
-                llm_pick = openai_final_pick(
-                    openai_api_key=openai_api_key.strip(),
-                    user_answers=[a for a in answers if a is not None],
-                    best_group=best_group,
-                    group_scores=group_scores,
-                    movies=movies,
-                    model=openai_model.strip(),
-                )
-        except Exception as e:
-            llm_pick = {"error": str(e)}
-
-    st.session_state.result_ready = True
-    st.session_state.best_group = best_group
-    st.session_state.movies = movies
-    st.session_state.llm_pick = llm_pick
-
-# 결과 화면 렌더
-if st.session_state.result_ready:
-    best_group = st.session_state.best_group
-    movies = st.session_state.movies
-    llm_pick = st.session_state.llm_pick
-
-    # 1) 결과 제목
-    st.markdown(f"## ✨ 당신에게 딱인 장르는: **{best_group}**!")
-
-    # (추가) LLM 최종 1편 추천
-    if llm_pick:
-        st.divider()
-        st.subheader("🤖 OpenAI 최종 1픽")
-
-        if isinstance(llm_pick, dict) and llm_pick.get("error"):
-            st.warning("OpenAI 최종 추천을 불러오지 못했어. (키/모델/네트워크 확인)")
-            st.caption(llm_pick["error"])
-        else:
-            pick_id = llm_pick.get("movie_id")
-            pick_title = llm_pick.get("title")
-            pick_reason = llm_pick.get("reason", "")
-            pick_conf = llm_pick.get("confidence", None)
-
-            # TMDB 후보에서 해당 영화 찾기
-            picked_movie = next((m for m in movies if m.get("id") == pick_id), None)
-
-            c1, c2 = st.columns([1, 2], gap="large")
-            with c1:
-                if picked_movie and picked_movie.get("poster_path"):
-                    st.image(f"{POSTER_BASE}{picked_movie['poster_path']}", use_container_width=True)
-                else:
-                    st.info("포스터 없음 🖼️")
-            with c2:
-                st.markdown(f"### ⭐ 오늘의 최종 추천: {pick_title}")
-                if picked_movie and picked_movie.get("vote_average") is not None:
-                    st.write(f"평점: **{float(picked_movie['vote_average']):.1f} / 10**")
-                if pick_conf is not None:
-                    try:
-                        st.caption(f"신뢰도(모델 추정): {float(pick_conf):.2f}")
-                    except Exception:
-                        pass
-                st.markdown("**추천 이유**")
-                st.write(pick_reason)
-
-    st.divider()
-
-    # 2) 영화 카드 3열 표시 + 3) 포스터/제목/평점 + 4) expander 상세 + 5) spinner(상세 로딩)
-    st.subheader("🍿 TMDB 추천 영화 5편 (카드)")
-    cols = st.columns(3, gap="large")
-
-    for idx, m in enumerate(movies):
-        col = cols[idx % 3]
-        movie_id = m.get("id")
-        title = m.get("title") or m.get("original_title") or "제목 없음"
-        rating = m.get("vote_average")
-        overview = safe_text(m.get("overview"))
-        poster_path = m.get("poster_path")
-        poster_url = f"{POSTER_BASE}{poster_path}" if poster_path else None
-
-        with col:
-            with st.container(border=True):
-                if poster_url:
-                    st.image(poster_url, use_container_width=True)
-                else:
-                    st.info("포스터 없음 🖼️")
-
-                st.markdown(f"### {title}")
-                if rating is not None:
-                    st.write(f"⭐ **{float(rating):.1f} / 10**")
-                else:
-                    st.write("⭐ 평점 정보 없음")
-
-                # 카드 클릭(확장) -> 상세 정보
-                with st.expander("상세 정보 보기"):
-                    with st.spinner("상세 정보를 불러오는 중..."):
-                        details = {}
-                        try:
-                            details = tmdb_movie_details(tmdb_api_key.strip(), movie_id, language)
-                        except Exception:
-                            details = {}
-
-                    st.markdown("**줄거리**")
-                    st.write(overview if overview else "줄거리 정보가 없어요.")
-
-                    # 키워드(있으면)
-                    kw_obj = details.get("keywords", {})
-                    if isinstance(kw_obj, dict):
-                        kws = [k.get("name") for k in kw_obj.get("keywords", []) if k.get("name")]
-                        if kws:
-                            st.markdown("**키워드**")
-                            st.write(", ".join(kws[:10]))
-
-    st.caption("※ TMDB는 작품/언어별로 줄거리 데이터가 비어 있을 수 있어요. (ko-KR 비어 있으면 en-US로 바꿔보기)")
+from __future__ import annotations
+
+import streamlit as st
+
+
+st.set_page_config(page_title="Y-Compass 기획서", page_icon="🧭", layout="wide")
+
+st.title("🧭 Y-Compass (와이컴퍼스)")
+st.caption("연세대 AX 캠프 트랙1 - 소그룹 챌린지 | 나만의 앱 기획서")
+
+st.markdown(
+    """
+**한줄 소개**  
+대학 진학이 막막한 10대에게, 근거 있는 전형/전공 후보 3개 + 8주 준비 로드맵을 뽑아주는 AI 진학 카운셀러
+"""
+)
+
+st.divider()
+
+with st.sidebar:
+    st.header("📌 빠른 이동")
+    st.markdown(
+        """
+- [1. 개요](#1-개요)
+- [2. 핵심 기능](#2-핵심-기능-3가지-이내)
+- [3. 화면 구성](#3-화면-구성)
+- [4. AI 활용 계획](#4-ai-활용-계획)
+- [5. 예상 프롬프트](#5-예상-프롬프트-요약)
+- [6. 참고/영감](#6-참고영감)
+- [7. 피드백 반영](#7-피드백-반영--보완-과제)
+"""
+    )
+
+    st.divider()
+    st.subheader("✅ 목표 사용자")
+    st.write("수시/정시를 고민하며 전공·전형 선택이 막막한 고3·N수생")
+
+
+st.markdown("## 1. 개요")
+st.markdown(
+    """
+### 해결하고 싶은 문제
+- 정보가 너무 많고 흩어져 있어 무엇부터 봐야 할지 모름  
+- 전문 컨설팅은 비용 부담 + 지역/시간 제약으로 접근성 낮음  
+- “카더라” 조언이 많아 근거/출처가 불투명함
+
+### 해결 전략
+짧은 입력 → 근거 기반 추천(출처 제시) → 바로 실행 가능한 로드맵으로 전환  
+사교육비가 늘어나는 현실에서 **낮은 비용/높은 접근성**의 가치가 더 선명해짐
+"""
+)
+
+st.markdown("## 2. 핵심 기능 (3가지 이내)")
+
+col1, col2, col3 = st.columns(3, gap="large")
+with col1:
+    st.subheader("기능 1: 8문항 ‘진학 상황 스캔’")
+    st.markdown(
+        """
+- 입력: 성적대(범주), 관심 전공군, 선호 활동/적성, 비교과/동아리/수상 유무  
+- 제약(지역·예산·시간), 목표 우선순위  
+- 출력: 상황 5줄 요약 + 핵심 제약/강점 3개
+"""
+    )
+
+with col2:
+    st.subheader("기능 2: 후보 3개 추천")
+    st.markdown(
+        """
+- A: 안정 / B: 적정 / C: 도전 (3경로 고정)  
+- 추천 이유 3줄 + 준비 액션 5개  
+- 근거/출처 요약 (RAG 기반)
+"""
+    )
+
+with col3:
+    st.subheader("기능 3: 8주 준비 로드맵")
+    st.markdown(
+        """
+- 0–2주 / 3–5주 / 6–8주 구간화  
+- 체크리스트 + 리소스  
+- 산출물: 자소서 초안, 포트폴리오, 활동기록  
+- PDF 내보내기 또는 플랜 저장
+"""
+    )
+
+st.markdown("## 3. 화면 구성")
+screen_col1, screen_col2 = st.columns(2, gap="large")
+
+with screen_col1:
+    st.subheader("메인 화면 (입력/진단)")
+    st.markdown(
+        """
+- 온보딩: “3분이면 방향 잡아요”  
+- 8문항 입력(라디오/멀티셀렉트 중심)  
+- 개인정보 최소화 + “추천은 참고용” 고지
+"""
+    )
+
+with screen_col2:
+    st.subheader("결과/출력 화면")
+    st.markdown(
+        """
+- 내 상황 요약(5줄) + 강점/제약 태그  
+- 후보 3개 카드(A/B/C) + 적합도(낮/중/높)  
+- 추천 이유 3줄 + 준비 액션 5개  
+- 8주 로드맵 타임라인/체크리스트  
+- 근거 보기(출처) 아코디언
+"""
+    )
+
+st.markdown("## 4. AI 활용 계획")
+st.markdown(
+    """
+1. **요약/진단**: 입력 → 상황 요약 + 핵심 변수 추출  
+2. **경로 생성**: 후보 3개(A/B/C) + 액션 리스트 생성  
+3. **로드맵 생성**: 8주 계획표/체크리스트 생성  
+4. **RAG (권장)**: 전형요강/학과소개/재학생 팁/공공 리소스 → 임베딩 검색  
+
+**목적**: “환각 줄이기” + “출처를 보여주는 상담”으로 신뢰 확보
+"""
+)
+
+st.markdown("## 5. 예상 프롬프트 (요약)")
+st.code(
+    """System
+너는 진학 컨설턴트다. 사용자의 목표를 구체화하고, 3가지 경로를 제안하되,
+불확실성과 가정을 명시하고, 근거(제공된 문서) 기반으로만 사실을 말하라.
+
+User (JSON)
+학년/성적대/관심전공/선호활동/제약/목표
+
+Output
+- 5줄 상황요약
+- A/B/C 경로 (추천이유 3줄 + 액션 5개 + 리스크 2개)
+- 8주 로드맵 (주차별 3개 task)
+- 근거 목록 (문서 제목/요약)
+""",
+    language="text",
+)
+
+st.markdown("## 6. 참고/영감")
+st.markdown(
+    """
+- **Naviance**: 학생-진로/대학 탐색 + AI 챗봇 사례  
+- **Scoir**: AI 기반 대학/커리어 플래닝 흐름  
+- **Education Week**: 실제 상담사가 AI를 쓰는 방식  
+- **Seoul Learn**: 교육 접근성(무료/공공 리소스)  
+- **Harvard Business Review**: 커리어 코칭 비용 문제의식
+"""
+)
+
+st.markdown("## 7. 피드백 반영 & 보완 과제")
+
+st.markdown("### Good 👍")
+st.markdown(
+    """
+- AI Prompting 설계가 정교하고 실제 입시 상담 프로토콜 느낌  
+- 희망 전형 입력란 + 수시/정시 세부 분기 반영이 깔끔함
+"""
+)
+
+st.markdown("### 고민할 부분")
+st.markdown(
+    """
+- 입시 데이터 출처 확보: 특정 대학(예: 연세대) 주요 학과 5~10개에 대해
+  최근 2~3년 입시 데이터를 정리 후 프롬프트에 반영  
+- API 활용 여부 검토 및 데이터가 없는 학교는 일반 전략 가이드 제공  
+- 성적(내신/모의고사) 추가 입력으로 안정/적정/도전 분석 정교화  
+- 수시/정시와 현재 시점을 반영한 주차별 핵심 목표 1개 + 할 일 2~3개 제시
+"""
+)
+
+st.markdown("### 동료 피드백 반영 핵심")
+st.markdown(
+    """
+- 사용자가 원하는 전형을 직접 선택하는 입력란 추가  
+- 수시/정시 대분류 + 수시 세부 전형 분기  
+- 과거 입시 결과 기반 진학 가능성 표시  
+- 상용화 시 가격 정책 및 요금제별 제공 범위 설계
+"""
+)
+
+st.divider()
+st.subheader("상세 정의 (Technical Spec)")
+spec_data = [
+    {
+        "구분": "Input Data",
+        "상세 정의": "사용자 희망 전형(직접 선택/입력) + 수시/정시 대분류 + 수시 세부 전형 분기",
+    },
+    {
+        "구분": "AI Prompting",
+        "상세 정의": (
+            "사용자가 선택한 전형을 존중하되, 가능성·리스크·대안까지 제시. "
+            "입력 전형이 있을 경우 해당 전형 우선 분석하고, 성적·비교과·전형 특성 충족 정도를 "
+            "근거와 함께 평가. 가능성이 낮을 경우 대체 가능한 전형(수시/정시 등) 전략을 제안. "
+            "과거 입시 결과는 기준 연도와 한계를 명시하고, 확률 단정 대신 구간(안정/적정/도전)으로 표현."
+        ),
+    },
+    {
+        "구분": "Output Format",
+        "상세 정의": (
+            "1) 사용자가 원하는 전형 가능성 표시 "
+            "2) AI 추천 전형/전략 TOP3 "
+            "3) 8주 로드맵"
+        ),
+    },
+]
+st.table(spec_data)
+
+st.caption("이 문서는 기획서 요약본입니다. 실제 서비스 구현 시 데이터 소싱 및 검증 체계가 필수입니다.")
 
EOF
)
