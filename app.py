# app_final_working_full_fixed_v3.py
# Run: streamlit run app_final_working_full_fixed_v3.py
# Requirements: pip install streamlit pandas numpy scikit-learn requests

import random
import requests
import re
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional

# -------------------------------
# Config (must be FIRST Streamlit call)
# -------------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

TMDB_API_KEY = "33309f53a02017934296c9319ac7b6f2"  # replace if you want
USE_TMDB = bool(TMDB_API_KEY)
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://via.placeholder.com/150x225?text=No+Poster"

# -------------------------------
# Helpers: title cleaning & variants
# -------------------------------
def strip_years_and_cleanup(title: str) -> str:
    """Remove all (YYYY) patterns and extra parentheses/quotes, collapse whitespace."""
    if not isinstance(title, str):
        title = str(title)
    cleaned = re.sub(r"\(\s*\d{4}\s*\)", "", title)
    cleaned = cleaned.replace("(", " ").replace(")", " ").replace('"', " ").replace("'", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def generate_search_variants(title: str, year: Optional[int]) -> list:
    """Produce a list of query strings to try against TMDB from most to least strict."""
    variants = []
    cleaned = strip_years_and_cleanup(title)
    if cleaned:
        variants.append(cleaned)
    m = re.match(r"^(.*),\s*(The|A|An)$", cleaned, flags=re.I)
    if m:
        variants.append(f"{m.group(2)} {m.group(1)}")
    simple = re.sub(r"[^\w\s]", " ", title)
    simple = re.sub(r"\s+", " ", simple).strip()
    if simple and simple not in variants:
        variants.append(simple)
    if title not in variants:
        variants.append(title)
    if year:
        yeared = variants[0] + f" {year}"
        if yeared not in variants:
            variants.insert(0, yeared)
    return variants

# -------------------------------
# Fetch TMDB details with fallbacks
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_movie_details(title: str, year: Optional[int] = None) -> Tuple[str, str, Optional[str]]:
    """Return (poster_url, overview, trailer_url)."""
    if not USE_TMDB:
        return PLACEHOLDER_POSTER, "Overview not available (TMDB disabled).", None

    variants = generate_search_variants(title, year)
    try:
        for query in variants:
            params = {"api_key": TMDB_API_KEY, "query": query}
            if year:
                params["year"] = int(year)
            resp = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=6)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results") or []
            if not results:
                continue
            candidate = results[0]
            if year:
                for r in results:
                    rd = r.get("release_date") or ""
                    if rd and rd.startswith(str(year)):
                        candidate = r
                        break
            poster_path = candidate.get("poster_path")
            poster_url = TMDB_IMAGE_BASE + poster_path if poster_path else PLACEHOLDER_POSTER
            overview = candidate.get("overview") or "Overview not available."
            tmdb_id = candidate.get("id")
            trailer_url = None
            if tmdb_id:
                try:
                    vresp = requests.get(
                        f"{TMDB_BASE_URL}/movie/{tmdb_id}/videos",
                        params={"api_key": TMDB_API_KEY},
                        timeout=6
                    )
                    vresp.raise_for_status()
                    vdata = vresp.json()
                    for video in vdata.get("results", []):
                        if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                            trailer_url = f"https://www.youtube.com/watch?v={video.get('key')}"
                            break
                except Exception:
                    trailer_url = None
            return poster_url, overview, trailer_url
        return PLACEHOLDER_POSTER, "Overview not available.", None
    except Exception as e:
        print(f"TMDB fetch error for '{title}': {e}")
        return PLACEHOLDER_POSTER, "Overview not available (TMDB error).", None

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data(movies_path: str = "movies.csv", ratings_path: str = "ratings.csv"):
    try:
        movies = pd.read_csv(movies_path)
    except Exception:
        movies = pd.DataFrame()
    try:
        ratings = pd.read_csv(ratings_path)
    except Exception:
        ratings = pd.DataFrame()

    if not movies.empty:
        if "genres" not in movies.columns:
            movies["genres"] = ""
        movies["genres"] = movies["genres"].fillna("").astype(str)
        movies["title"] = movies["title"].astype(str)
        if "movieId" in movies.columns:
            movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
        # ✅ Create year column
        movies["year"] = pd.to_numeric(
            movies["title"].str.extract(r"\((\d{4})\)", expand=False),
            errors="coerce"
        )

    if not ratings.empty and "movieId" in ratings.columns:
        ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")

    return movies, ratings

movies, ratings = load_data()

if movies.empty:
    st.error("No movie data available. Provide movies.csv to continue.")
    st.stop()

# -------------------------------
# TF-IDF model
# -------------------------------
genre_corpus = movies["genres"].fillna("")
if genre_corpus.str.len().sum() == 0:
    genre_corpus = movies["title"].fillna("")

tfidf = TfidfVectorizer(stop_words="english")
try:
    tfidf_matrix = tfidf.fit_transform(genre_corpus)
except Exception as e:
    st.error(f"Error building TF-IDF matrix: {e}")
    st.stop()

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ✅ Normalize titles for lookup
movies["title_norm"] = movies["title"].str.strip().str.lower()
indices = pd.Series(movies.index, index=movies["title_norm"]).drop_duplicates()

# -------------------------------
# Recommendation function
# -------------------------------
def recommend_content_based(title: str, n: int = 5, genre: Optional[str] = None,
                            year: Optional[str] = None, min_rating: Optional[float] = None):
    norm_title = title.strip().lower()
    if norm_title not in indices.index:
        return pd.DataFrame()
    idx = indices[norm_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: 1 + n * 10]
    movie_indices = [i for i, _ in sim_scores]

    recs = movies.iloc[movie_indices].copy()

    if not ratings.empty and "movieId" in ratings.columns:
        avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
        recs = recs.merge(avg_ratings, on="movieId", how="left").rename(columns={"rating": "avg_rating"})
    else:
        recs["avg_rating"] = pd.NA

    if genre:
        recs = recs[recs["genres"].str.contains(genre, regex=False, na=False)]
    if year and year.isdigit():
        recs = recs[recs["year"] == int(year)]
    if min_rating is not None:
        recs = recs[pd.to_numeric(recs["avg_rating"], errors="coerce") >= float(min_rating)]

    if "movieId" in recs.columns:
        recs = recs.drop_duplicates(subset=["movieId"])
    else:
        recs = recs.drop_duplicates(subset=["title"])

    return recs.head(n)[["title", "genres", "year", "avg_rating"]]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Movie Recommendation System")
st.write("Search for a movie, apply filters, or get a random suggestion!")

if "movie_choice" not in st.session_state:
    st.session_state.movie_choice = None

col1, col2 = st.columns([3, 1])
with col1:
    movie_list = movies["title"].dropna().sort_values().unique().tolist()
    selected_movie = st.selectbox("Search a movie:", movie_list, index=0)
    st.session_state.movie_choice = selected_movie
with col2:
    if st.button("🎲 Surprise Me!"):
        st.session_state.movie_choice = random.choice(movie_list)
        st.success(f"Surprise movie: {st.session_state.movie_choice}")

col1f, col2f, col3f = st.columns(3)
with col1f:
    all_genres = sorted({g for gs in movies["genres"].dropna().astype(str) for g in gs.split("|") if g})
    genre_filter = st.selectbox("Filter by Genre:", [""] + all_genres)
with col2f:
    years = sorted(movies["year"].dropna().unique().astype(int).tolist())
    year_filter = st.selectbox("Filter by Year:", [""] + [str(y) for y in years])
with col3f:
    min_rating = st.slider("Minimum Avg Rating:", 0.0, 5.0, 0.0, 0.5)

if st.session_state.movie_choice:
    results = recommend_content_based(
        st.session_state.movie_choice,
        n=5,
        genre=(genre_filter if genre_filter else None),
        year=(year_filter if year_filter else None),
        min_rating=(min_rating if min_rating > 0 else None),
    )
    if results.empty:
        st.warning("No recommendations found.")
    else:
        for _, row in results.iterrows():
            y = int(row["year"]) if pd.notna(row["year"]) else None
            poster_url, overview, trailer_url = fetch_movie_details(row["title"], y) if USE_TMDB else (PLACEHOLDER_POSTER, "Overview not available.", None)
            if not poster_url or isinstance(poster_url, (int, float)):
                poster_url = PLACEHOLDER_POSTER

            c1, c2 = st.columns([1, 3])
            with c1:
                st.image(poster_url, width=150)
            with c2:
                title_line = f"{row['title']}"
                if pd.notna(row.get('year')):
                    title_line += f" ({int(row['year'])})"
                st.subheader(title_line)
                st.write(f"**Genres:** {row['genres']}")
                rating_val = pd.to_numeric(row["avg_rating"], errors="coerce")
                st.write(f"**Avg Rating:** {round(rating_val, 2) if pd.notna(rating_val) else 'N/A'}")
                st.write(f"**Overview:** {overview}")
                if trailer_url:
                    st.video(trailer_url)

st.write("---")
