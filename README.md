# Movie Recommender System

A content-based movie recommender built with Streamlit. Enter a movie name
and get 5 similar movies with posters, directors, and cast.

## How it works
- Combines overview, genres, keywords, cast, and director into tags
- Stems words and vectorizes them with CountVectorizer
- Finds similar movies using cosine similarity

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Get a free API key from https://www.themoviedb.org/settings/api
4. Set it: `export TMDB_API_KEY=your_key` (Windows: `set TMDB_API_KEY=your_key`)
5. Run: `streamlit run app.py`

## Dataset
TMDB 5000 Movies and Credits datasets (from Kaggle).
