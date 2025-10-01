'''
Streamlit Data Science Project Template
'''

from apputil import *
import streamlit as st

st.set_page_config(page_title="Netflix Data Exploration", layout="wide")
st.title("Netflix Movie Ratings per Year")
st.markdown("This table shows counts of movies by their ratings for each year since 2016.")

styled_table = get_styled_rating_table()
st.dataframe(styled_table, use_container_width=True)
