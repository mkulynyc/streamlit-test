'''
Streamlit Data Science Project Template
'''

from apputil import *
import streamlit as st

st.set_page_config(page_title="Netflix Data Explorer", layout="wide")

st.title("🎬 Netflix Data Explorer")

# Load and clean data
df_raw = load_data()
df_clean, _ = cleanNetflixData(df_raw)

# Section 1: Ratings Table
st.header("📊 Movie Ratings per Year")
st.markdown("This table shows counts of movies by their ratings for each year since 2016.")
styled_table = get_styled_rating_table()
st.dataframe(styled_table, use_container_width=True)

# Section 2: Genre Trends
st.header("📈 Top Genres in the US Over Time")
st.markdown("Explore how the most popular genres have evolved over time for US-based Netflix titles.")

top_n = st.sidebar.slider("Select number of top genres", min_value=3, max_value=10, value=5)
fig = plot_top_us_genres(df_clean, top_n=top_n)
st.pyplot(fig)

