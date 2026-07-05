import streamlit as st
import numpy as np
import pandas as pd
import re
from rapidfuzz import process

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------

st.markdown("""
<style>

.stButton>button{
    width:100%;
    background-color:#E50914;
    color:white;
    border-radius:8px;
    font-weight:bold;
}

img{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Data
# -------------------------------------------------

@st.cache_data
def load_matrix():
    return pd.read_csv("movie_vectors1.csv", header=None).to_numpy(dtype=float)

@st.cache_data
def load_movies():
    return pd.read_csv("movies.csv")

vh = load_matrix()
movies = load_movies()

titles = movies["SearchTitle"].tolist()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

with st.sidebar:
    st.title("🎥 Movie Recommender")
    st.write("Built using:")
    st.write("- Singular Value Decomposition (SVD)")
    st.write("- Cosine Similarity")
    st.write(f"📽 Movies in database: **{len(movies)}**")

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def clean_title(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9 ]', '', title)
    return title.strip()


def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def find_recommendations(movie_index, k=10,include_selected=False):

    movie_vector = vh[:, movie_index]

    similarities = []

    for i in range(vh.shape[1]):

        if not include_selected and i == movie_index:
            continue

        sim = cosine_similarity(movie_vector, vh[:, i])

        similarities.append((sim, i))

    similarities.sort(reverse=True)

    return similarities[:k]


def search_movie(query):

    cleaned = clean_title(query)

    # Exact Match
    if cleaned in titles:
        return titles.index(cleaned), True

    # Closest Match
    match = process.extractOne(cleaned, titles)

    if match:
        return match[2], False

    return None, False


# -------------------------------------------------
# UI
# -------------------------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Find movies similar to your favorite films using "
    "**Singular Value Decomposition (SVD)** and **Cosine Similarity**."
)

query = st.text_input(
    "Search a Movie",
    placeholder="Example: Interstellar"
)

if st.button("Recommend Movies"):

    if query.strip() == "":
        st.warning("Please enter a movie name.")
        st.stop()

    movie_index, exact_match = search_movie(query)

    if movie_index is None:
        st.error("No similar movie could be found.")
        st.stop()

    # -----------------------------------------
    # Selected Movie
    # -----------------------------------------

    if exact_match:

        movie = movies.iloc[movie_index]

        st.subheader("🎬 Selected Movie")

        left, right = st.columns([1, 3])

        with left:

            st.image(movie["POSTER_URL"], use_container_width=True)

        with right:

            st.markdown(f"## {movie['Title']}")

            st.write(f"**Year:** {movie['Year']}")

            st.write(f"**Genres:** {movie['Genres']}")

        st.divider()

    else:

        closest_movie = movies.iloc[movie_index]["Title"]

        st.warning(
            f"**'{query}'** is not available in our movie database.\n\n"
            f"Showing recommendations based on the closest available movie: **{closest_movie}**."
        )

    # -----------------------------------------
    # Recommendations
    # -----------------------------------------

    st.subheader("🍿 Recommended Movies")

    recommendations = find_recommendations(movie_index, k=10,include_selected=not exact_match)

    cols = st.columns(5)

    for idx, (score, movie_id) in enumerate(recommendations):

        rec = movies.iloc[movie_id]

        with cols[idx % 5]:

            st.image(
                rec["POSTER_URL"],
                use_container_width=True
            )

            st.markdown(
                f"**{rec['Title']} ({rec['Year']})**"
            )

            st.caption(rec["Genres"])

            st.progress(float(score))

            st.write(f"**Match:** {score*100:.1f}%")