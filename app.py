import ast

import numpy as np
import pandas as pd
import requests
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Keep both CSV files in the same folder as app.py.
credits = pd.read_csv("tmdb_5000_credits.csv", engine="python", quotechar='"')
movies_df = pd.read_csv("tmdb_5000_movies.csv", engine="python", quotechar='"')

movies_df = movies_df.merge(credits, on="title")

# Clean and select columns
movies_df = movies_df[
    ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
]
movies_df.dropna(inplace=True)


# Convert JSON-like strings to lists
def convert(obj):
    return [item["name"] for item in ast.literal_eval(obj)]


# Get the top 3 cast members
def convert3(obj):
    cast = []
    for item in ast.literal_eval(obj):
        if len(cast) == 3:
            break
        cast.append(item["name"])
    return cast


# Get the director
def fetch_director(obj):
    for item in ast.literal_eval(obj):
        if item["job"] == "Director":
            return [item["name"]]
    return []


# Keep cast and director names for display
movies_df["cast_display"] = movies_df["cast"].apply(convert3)
movies_df["director_display"] = movies_df["crew"].apply(fetch_director)

# Process fields for recommendations
movies_df["genres"] = movies_df["genres"].apply(convert)
movies_df["keywords"] = movies_df["keywords"].apply(convert)
movies_df["cast"] = movies_df["cast"].apply(convert3)
movies_df["crew"] = movies_df["crew"].apply(fetch_director)
movies_df["overview"] = movies_df["overview"].apply(lambda text: text.split())

# Remove spaces from multi-word tags
for column in ["genres", "keywords", "cast", "crew"]:
    movies_df[column] = movies_df[column].apply(
        lambda items: [item.replace(" ", "") for item in items]
    )

# Create the tags used by the recommendation system
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

new_df["tags"] = new_df["tags"].apply(lambda items: " ".join(items).lower())

# Stem words
ps = PorterStemmer()


def stem(text):
    return " ".join(ps.stem(word) for word in text.split())


new_df["tags"] = new_df["tags"].apply(stem)

# Vectorize tags and calculate similarity
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()
similarity = cosine_similarity(vectors)


# TMDB API key for fetching posters
API_KEY = "YOUR_TMDB_API_KEY"


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
                return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except requests.RequestException:
        return None

    return None


def recommend_with_details(movie):
    matching_movies = new_df["title"].str.lower() == movie.lower()

    if not matching_movies.any():
        return [], [], [], []

    movie_index = new_df.index[matching_movies][0]
    distances = similarity[movie_index]

    movie_list = sorted(
        enumerate(distances),
        reverse=True,
        key=lambda item: item[1],
    )[1:6]

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


# Streamlit interface
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender System")
st.write("Type a movie name to get recommendations with posters.")

movie_input = st.text_input("Enter a movie name")

if movie_input:
    titles, posters, cast_lists, directors = recommend_with_details(movie_input)

    if not titles:
        st.warning("Movie not found. Try a different title.")
    else:
        columns = st.columns(5)

        for index, column in enumerate(columns):
            with column:
                st.subheader(titles[index])

                if posters[index]:
                    st.image(posters[index], use_container_width=True)
                else:
                    st.text("(No poster found)")

                if directors[index]:
                    st.markdown(f"**Director:** {', '.join(directors[index])}")

                if cast_lists[index]:
                    st.markdown(f"**Cast:** {', '.join(cast_lists[index])}")
