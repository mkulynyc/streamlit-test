"""
Utility functions for Netflix data cleaning and table generation.
Compatible with Python 3.10 and Streamlit Cloud.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def add_ratings(df):
    ### -------- Step 1: Load IMDB Datasets -------- ###

    # Load titles
    basics = pd.read_csv(
        "title.basics.tsv",
        sep="\t",
        na_values="\\N",
        usecols=["tconst", "primaryTitle", "originalTitle", "startYear", "titleType"]
    )
    # Load ratings
    ratings = pd.read_csv(
        "title.ratings (1).tsv",
        sep="\t",
        na_values="\\N"
    )


    ### -------- Step 2: Merge Datasets -------- ###

    imdb = basics.merge(ratings, on="tconst", how="inner")


    ### -------- Step 3: Filter to Movies Only -------- ###

    imdb_movies = imdb[imdb['titleType'] == "movie"].copy()


    ### -------- Step 4: Merge with Netflix Movies -------- ###

    # Uses clean movies data using Aidan's cleaning function, filter so type is movie
    movie_ratings_data = df.merge(
        imdb_movies,
        how = "inner",
        left_on="title",
        right_on = "primaryTitle"
    )

    ### -------- Step 5: Drop Duplicate Movies -------- ###

    # Sort the dataframe so that for each title, the row with the highest numVotes comes first
    movie_ratings_data_sorted = movie_ratings_data.sort_values(by='numVotes', ascending=False)

    # Drop duplicates based on 'title', keeping the first (which has the highest numVotes)
    movie_ratings_data_deduped = movie_ratings_data_sorted.drop_duplicates(subset='title', keep='first')

    # Optional: reset the index
    movie_ratings_data_deduped = movie_ratings_data_deduped.reset_index(drop=True)
    
    return(movie_ratings_data_deduped)

    

# Data cleaning function
def cleanNetflixData(df,
                     estimateSeasonMinutes=False,
                     episodesPerSeason=10,
                     minutesPerEpisode=45,
                     explodeGenres=False,
                     standardizeGenres=True):

    df = df.copy()

    # Trim whitespace and normalize empties
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    # Parse date_added
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["year_added"] = df["date_added"].dt.year.astype("Int64")
        df["month_added"] = df["date_added"].dt.month.astype("Int64")

    # Normalize type
    if "type" in df.columns:
        df["type"] = df["type"].str.title().replace({"Tv Show": "TV Show"})

    # Normalize rating
    if "rating" in df.columns:
        r = df["rating"].str.upper().str.replace(" ", "-", regex=False)
        r = r.replace({
            "TVMA": "TV-MA", "TV14": "TV-14", "TVPG": "TV-PG", "TVG": "TV-G",
            "TVY7": "TV-Y7", "TVY": "TV-Y"
        })
        df["rating"] = r

    # Extract duration in minutes and seasons
    df["duration_minutes"] = pd.to_numeric(
        df.get("duration", pd.Series(index=df.index, dtype="object"))
          .str.extract(r"(\d+)\s*min", expand=False),
        errors="coerce"
    ).astype("Int64")

    df["seasons"] = pd.to_numeric(
        df.get("duration", pd.Series(index=df.index, dtype="object"))
          .str.extract(r"(?i)(\d+)\s*Season", expand=False),
        errors="coerce"
    ).astype("Int64")

    # Estimate TV show duration
    if estimateSeasonMinutes:
        est = df["seasons"].astype("Float64") * episodesPerSeason * minutesPerEpisode
        df["duration_minutes"] = df["duration_minutes"].astype("Float64")
        df["duration_minutes"] = df["duration_minutes"].fillna(est).round().astype("Int64")

    # Country parsing
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")
        df["countries"] = df["country"].str.split(r"\s*,\s*")
        df["primary_country"] = df["countries"].apply(lambda xs: xs[0] if isinstance(xs, list) and len(xs) else "Unknown")

    # Fill missing director/cast
    for c in ["director", "cast"]:
        if c in df.columns:
            df[f"has_{c}"] = df[c].notna()
            df[c] = df[c].fillna("Unknown")

    # Genre parsing
    if "listed_in" in df.columns:
        df["genres"] = df["listed_in"].fillna("Unknown").str.split(r"\s*,\s*")
        if standardizeGenres:
            df["genres"] = df["genres"].apply(lambda genre_list: [
                re.sub(r'\s*(TV Shows?|Movies?|Series?|Dramas?)$', '', genre, flags=re.IGNORECASE).strip()
                for genre in genre_list
            ])
    else:
        df["genres"] = [[] for _ in range(len(df))]

    genres_exploded = None
    if explodeGenres:
        genres_exploded = (
            df[["show_id", "title", "genres"]]
            .explode("genres", ignore_index=False)
            .rename(columns={"genres": "genre"})
            .reset_index(drop=False)
            .rename(columns={"index": "row_idx"})
        )

    # Release year validation
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
        current_year = pd.Timestamp('now').year
        df.loc[df['release_year'] > current_year + 1, 'release_year'] = pd.NA
        df.loc[df['release_year'] < 1900, 'release_year'] = pd.NA

    df = df.drop_duplicates(subset=["show_id"]).reset_index(drop=True)

    # Flags
    df["is_movie"] = (df.get("type", "") == "Movie")
    df["is_tv"] = (df.get("type", "") == "TV Show")

    # Release month/day
    if "release_year" in df.columns:
        temp_date = pd.to_datetime(df['release_year'], format='%Y', errors='coerce')
        df['release_month'] = temp_date.dt.month.astype('Int64')
        df['release_day'] = temp_date.dt.day.astype('Int64')

    return (df, genres_exploded) if explodeGenres else (df, None)


# Load dataset
@st.cache_data
def load_data(file_path="./netflix_titles.csv"):
    df = pd.read_csv(file_path)
    df, _ = cleanNetflixData(df)
    df = add_ratings(df)
    return df

# Movie recommendation functions
def keyword_match(df, keywords, match_mode='any'):
    if match_mode == 'all':
        return df[df['description'].apply(
            lambda desc: all(re.search(re.escape(k), desc, re.IGNORECASE) for k in keywords)
        )]
    else:
        pattern = '|'.join([re.escape(k) for k in keywords])
        return df[df['description'].str.contains(pattern, case=False, na=False)]

def genre_filter(df, genre_list, match_mode='any'):
    genre_list_lower = [g.lower() for g in genre_list]

    def match_genres(g):
        g_lower = [x.lower() for x in g if x]
        if match_mode == 'all':
            return all(genre in g_lower for genre in genre_list_lower)
        else:
            return any(genre in g_lower for genre in genre_list_lower)

    return df[df['genres_list'].apply(match_genres)]

def recommend_movies(df, keywords=None, genres=None, top_n=10, keyword_match_mode='any', genre_match_mode='any'):
    filtered = df.copy()

    if keywords:
        filtered = keyword_match(filtered, keywords, match_mode=keyword_match_mode)

    if genres:
        filtered = genre_filter(filtered, genres, match_mode=genre_match_mode)

    if filtered.empty:
        return pd.DataFrame(columns=['title', 'averageRating', 'genres', 'description'])

    return (
        filtered
        .sort_values(by='averageRating', ascending=False)
        .loc[:, ['title', 'averageRating', 'genres', 'description']]
        .head(top_n)
    )



