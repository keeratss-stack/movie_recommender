import ast
import html
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,500..700;1,6..96,500&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --night: #0F2327;
  --panel: #16343A;
  --raised: #1D424A;
  --paper: #EDF3EE;
  --muted: #8FAAAE;
  --lamp: #F4B14E;
  --line: rgba(237, 243, 238, 0.14);
  --display: 'Bodoni Moda', Georgia, 'Times New Roman', serif;
  --body: 'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

.stApp { background: var(--night); font-family: var(--body); }
.stApp [data-testid="stHeader"] { background: transparent; }
.stApp [data-testid="stDecoration"], #MainMenu, footer { display: none; }
.stApp .block-container { max-width: 1180px; padding: 2.5rem 1.5rem 4rem; }

.stApp .hero { position: relative; padding: 2rem 0 1.4rem; }
.stApp .hero::before {
  content: ""; position: absolute; z-index: 0; pointer-events: none;
  inset: -140px -10% auto 25%; height: 440px;
  background: radial-gradient(55% 60% at 50% 0%, rgba(244, 177, 78, 0.20), transparent 72%);
}
.stApp .hero h1 {
  position: relative; z-index: 1; margin: 0; padding: 0;
  font-family: var(--display); font-weight: 600; letter-spacing: -0.01em;
  font-size: clamp(2.3rem, 5.6vw, 4.2rem); line-height: 1.04;
  color: var(--paper); max-width: 15ch;
}
.stApp .hero p {
  position: relative; z-index: 1; margin: 1rem 0 0; max-width: 52ch;
  font-size: 1.1rem; line-height: 1.55; color: var(--muted);
}

.stApp [data-baseweb="select"] > div {
  min-height: 3.5rem; border-radius: 6px; font-size: 1.05rem;
  background: var(--panel); border: 1px solid var(--line);
}
.stApp [data-baseweb="select"] > div:focus-within {
  border-color: var(--lamp); box-shadow: 0 0 0 1px var(--lamp);
}

.stApp .quick-label { margin: 1rem 0 0.4rem; font-size: 0.92rem; color: var(--muted); }
.stApp [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem; }
.stApp [data-testid="stColumn"], .stApp [data-testid="column"] {
  width: auto !important; flex: 0 0 auto !important; min-width: 0 !important;
}
.stApp .stButton > button {
  border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.92rem;
  background: transparent; color: var(--paper); border: 1px solid var(--line);
  transition: border-color 0.2s, color 0.2s;
}
.stApp .stButton > button:hover,
.stApp .stButton > button:focus-visible { border-color: var(--lamp); color: var(--lamp); }

.stApp .pick {
  margin-top: 2.4rem; padding: 0.2rem 0 0.2rem 1.2rem;
  border-left: 2px solid var(--lamp);
}
.stApp .pick h2 {
  margin: 0; padding: 0; font-family: var(--display); font-weight: 600;
  font-size: clamp(1.5rem, 3vw, 2.1rem); line-height: 1.15; color: var(--paper);
}
.stApp .pick .pick-meta { margin: 0.45rem 0 0; font-size: 0.98rem; color: var(--muted); }

.stApp .grid {
  display: grid; gap: 2rem 1.3rem; margin-top: 1.8rem;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.stApp .card { display: flex; flex-direction: column; gap: 0.75rem; min-width: 0; }
.stApp .poster {
  display: block; width: 100%; aspect-ratio: 2 / 3; object-fit: cover;
  border-radius: 4px; background: var(--panel);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.stApp .card:hover .poster { transform: translateY(-4px); box-shadow: 0 16px 32px rgba(0, 0, 0, 0.45); }
.stApp .poster-blank {
  display: flex; align-items: center; justify-content: center; text-align: center;
  padding: 1.1rem; font-family: var(--display); font-size: 1.2rem; line-height: 1.2;
  color: var(--paper); background: linear-gradient(160deg, var(--raised), var(--panel));
  box-shadow: inset 0 0 0 1px rgba(244, 177, 78, 0.35);
}
.stApp .card h3 {
  margin: 0; padding: 0; font-family: var(--display); font-weight: 600;
  font-size: 1.12rem; line-height: 1.2; color: var(--paper);
}
.stApp .card .line { margin: 0; font-size: 0.9rem; line-height: 1.45; color: var(--muted); }
.stApp .card .line b { font-weight: 500; color: var(--paper); }

.stApp .sim { display: flex; align-items: center; gap: 0.6rem; }
.stApp .sim-bar { flex: 1; height: 3px; border-radius: 2px; background: var(--line); overflow: hidden; }
.stApp .sim-bar span { display: block; height: 100%; background: var(--lamp); }
.stApp .sim-label { font-size: 0.82rem; font-weight: 500; color: var(--lamp); white-space: nowrap; }

.stApp .foot {
  margin-top: 4rem; padding-top: 1.2rem; border-top: 1px solid var(--line);
  font-size: 0.82rem; line-height: 1.5; color: var(--muted); max-width: 70ch;
}

@media (prefers-reduced-motion: reduce) {
  .stApp .poster, .stApp .stButton > button { transition: none; }
  .stApp .card:hover .poster { transform: none; }
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


def find_file(*names):
    here = Path(__file__).parent
    for folder in (here, here / "data"):
        for name in names:
            for suffix in ("", ".gz"):
                path = folder / (name + suffix)
                if path.exists() and path.stat().st_size > 0:
                    return path
    return None


def convert(obj):
    return [item["name"] for item in ast.literal_eval(obj)]


def convert3(obj):
    cast = []
    for item in ast.literal_eval(obj):
        if len(cast) == 3:
            break
        cast.append(item["name"])
    return cast


def fetch_director(obj):
    for item in ast.literal_eval(obj):
        if item["job"] == "Director":
            return [item["name"]]
    return []


ps = PorterStemmer()


def stem(text):
    return " ".join(ps.stem(word) for word in text.split())


@st.cache_resource(show_spinner="Loading the movie library...")
def load_data():
    credits_path = find_file("tmdb_5000_credits.csv", "tmbd_5000_credits.csv")
    movies_path = find_file("tmdb_5000_movies.csv")
    if credits_path is None or movies_path is None:
        return None, None

    credits = pd.read_csv(credits_path, engine="python", quotechar='"')
    movies_df = pd.read_csv(movies_path, engine="python", quotechar='"')

    movies_df = movies_df.merge(credits, on="title")

    movies_df = movies_df[
        ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ]
    movies_df.dropna(inplace=True)

    movies_df["cast_display"] = movies_df["cast"].apply(convert3)
    movies_df["director_display"] = movies_df["crew"].apply(fetch_director)

    movies_df["genres"] = movies_df["genres"].apply(convert)
    movies_df["keywords"] = movies_df["keywords"].apply(convert)
    movies_df["cast"] = movies_df["cast"].apply(convert3)
    movies_df["crew"] = movies_df["crew"].apply(fetch_director)
    movies_df["overview"] = movies_df["overview"].apply(lambda text: text.split())

    for column in ["genres", "keywords", "cast", "crew"]:
        movies_df[column] = movies_df[column].apply(
            lambda items: [item.replace(" ", "") for item in items]
        )

    movies_df["tags"] = (
        movies_df["overview"]
        + movies_df["genres"]
        + movies_df["keywords"]
        + movies_df["cast"]
        + movies_df["crew"]
    )

    new_df = movies_df[
        ["movie_id", "title", "tags", "cast_display", "director_display"]
    ].copy()

    new_df = new_df.drop_duplicates(subset="title").reset_index(drop=True)

    new_df["tags"] = new_df["tags"].apply(lambda items: " ".join(items).lower())
    new_df["tags"] = new_df["tags"].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return new_df, similarity


new_df, similarity = load_data()

if new_df is None:
    st.error(
        "Could not find the data files. Put tmdb_5000_movies.csv and "
        "tmdb_5000_credits.csv (or the .gz versions) next to app.py or in a data folder."
    )
    st.stop()


API_KEY = "YOUR_TMDB_API_KEY"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_poster(title):
    if not API_KEY or API_KEY == "YOUR_TMDB_API_KEY":
        return None

    url = "https://api.themoviedb.org/3/search/movie"
    try:
        response = requests.get(
            url,
            params={"api_key": API_KEY, "query": title},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.RequestException:
        return None

    return None


def top_matches(movie):
    matching_movies = new_df["title"].str.lower() == movie.lower()

    if not matching_movies.any():
        return []

    movie_index = new_df.index[matching_movies][0]
    ranked = sorted(
        enumerate(similarity[movie_index]),
        reverse=True,
        key=lambda item: item[1],
    )
    return [item for item in ranked if item[0] != movie_index][:5]


def recommend_with_details(movie):
    movie_list = top_matches(movie)

    if not movie_list:
        return [], [], [], []

    titles = []
    posters = []
    cast_lists = []
    directors = []

    for index, _score in movie_list:
        movie_data = new_df.iloc[index]
        title = movie_data["title"]

        titles.append(title)
        posters.append(fetch_poster(title))
        cast_lists.append(movie_data["cast_display"])
        directors.append(movie_data["director_display"])

    return titles, posters, cast_lists, directors


def match_scores(movie):
    return [score for _, score in top_matches(movie)]


def esc(text):
    return html.escape(" ".join(str(text).split())).replace("$", "&#36;")


def squeeze(markup):
    return "".join(line.strip() for line in markup.splitlines())


def card_html(title, poster, cast, director, score):
    name = esc(title)
    if poster:
        image = f'<img class="poster" src="{esc(poster)}" alt="Poster for {name}" loading="lazy">'
    else:
        image = f'<div class="poster poster-blank" role="img" aria-label="{name}"><span>{name}</span></div>'

    pct = max(0, min(100, round(float(score) * 100)))
    director_line = (
        f'<p class="line">Directed by <b>{esc(", ".join(director))}</b></p>' if director else ""
    )
    cast_line = (
        f'<p class="line">With {esc(", ".join(cast))}</p>' if cast else ""
    )

    return squeeze(
        f"""
        <article class="card">
          {image}
          <h3>{name}</h3>
          <div class="sim"><div class="sim-bar"><span style="width:{pct}%"></span></div>
          <span class="sim-label">{pct}% alike</span></div>
          {director_line}{cast_line}
        </article>
        """
    )


def choose(title):
    st.session_state["movie"] = title


all_titles = sorted(new_df["title"].unique(), key=str.lower)
title_set = set(all_titles)
popular = [
    "Avatar",
    "The Dark Knight",
    "Inception",
    "Titanic",
    "Toy Story",
    "The Godfather",
    "Interstellar",
    "Iron Man",
]
quick_picks = [title for title in popular if title in title_set][:6]

st.markdown(
    squeeze(
        f"""
        <div class="hero">
          <h1>Find the film you’ll watch tonight.</h1>
          <p>Search {len(all_titles):,} movies, pick one you already love, and get
          five more that feel like it.</p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

movie_input = st.selectbox(
    "Search for a movie",
    all_titles,
    index=None,
    placeholder="Start typing a movie title...",
    key="movie",
    label_visibility="collapsed",
)

if quick_picks:
    st.markdown('<p class="quick-label">Or start with a popular one:</p>', unsafe_allow_html=True)
    for column, title in zip(st.columns(len(quick_picks)), quick_picks):
        with column:
            st.button(title, key=f"quick_{title}", on_click=choose, args=(title,))

if movie_input:
    titles, posters, cast_lists, directors = recommend_with_details(movie_input)

    if not titles:
        st.warning("Movie not found. Try a different title.")
    else:
        picked = new_df[new_df["title"].str.lower() == movie_input.lower()].iloc[0]
        picked_director = (
            f'<p class="pick-meta">Directed by {esc(", ".join(picked["director_display"]))}</p>'
            if picked["director_display"]
            else ""
        )
        st.markdown(
            squeeze(
                f"""
                <div class="pick">
                  <h2>Because you picked {esc(picked["title"])}</h2>
                  {picked_director}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        scores = match_scores(movie_input)
        cards = [
            card_html(titles[i], posters[i], cast_lists[i], directors[i], scores[i])
            for i in range(len(titles))
        ]
        st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)

st.markdown(
    squeeze(
        """
        <div class="foot">Movie data from TMDB. This product uses the TMDB API
        but is not endorsed or certified by TMDB.</div>
        """
    ),
    unsafe_allow_html=True,
)
