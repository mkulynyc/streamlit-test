# app.py

import streamlit as st
from apputil import load_data, recommend_movies

# Load and cache data
@st.cache_data
def get_data():
    return load_data()

df = get_data()

# Extract genres
all_genres = sorted(set(g for sublist in df['genres_list'] for g in sublist if g))

# UI
st.title("🎬 Movie Recommender")
st.markdown("**Available Genres:** " + ", ".join(all_genres))

keywords_input = st.text_input("Enter keywords (comma-separated)", value="school")
selected_genres = st.multiselect("Select genres", options=all_genres, default=["Classic"])

col1, col2 = st.columns(2)
with col1:
    keyword_mode = st.radio("Keyword match mode", ["any", "all"], horizontal=True)
with col2:
    genre_mode = st.radio("Genre match mode", ["any", "all"], horizontal=True)

top_n = st.slider("Number of recommendations", 1, 20, 10)

if st.button("Get Recommendations"):
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
    results = recommend_movies(df, keywords, selected_genres, top_n,
                               keyword_match_mode=keyword_mode,
                               genre_match_mode=genre_mode)

    if results.empty:
        st.warning("No matches found.")
    else:
        st.dataframe(results.reset_index(drop=True), use_container_width=True)