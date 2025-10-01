"""
Utility functions for Netflix data cleaning and table generation.
Compatible with Python 3.10 and Streamlit Cloud.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load dataset
def load_data(file_path="./netflix_titles.csv"):
    return pd.read_csv(file_path)

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

# Generate styled ratings table
def get_styled_rating_table():
    df = load_data()
    df_clean, _ = cleanNetflixData(df)
    df_movies = df_clean[df_clean['type'] == 'Movie'].copy()

    df_movies_ratings = df_movies.dropna(subset=['release_year', 'rating'])
    df_movies_ratings = df_movies_ratings[df_movies_ratings['release_year'] >= 2016]

    rating_counts = df_movies_ratings.groupby(['release_year', 'rating']).size().unstack(fill_value=0)

    styled_table = rating_counts.style.background_gradient(
        cmap='YlGnBu', axis=None
    ).set_caption("Movie Ratings Count per Year")

    return styled_table

# Plot top N genres in the US over time
def plot_top_us_genres(df_clean, top_n=5):
    # Filter US projects with valid genres and release year
    df_us = df_clean[df_clean['country'] == 'United States'].dropna(subset=['listed_in', 'release_year'])

    # Split genres into separate rows
    df_us_genres = df_us.assign(genre=df_us['listed_in'].str.split(', ')).explode('genre')

    # Count occurrences by year
    genre_counts = df_us_genres.groupby(['release_year', 'genre']).size().reset_index(name='count')

    # Get top N genres overall
    top_genres = genre_counts.groupby('genre')['count'].sum().sort_values(ascending=False).head(top_n).index
    genre_counts_top = genre_counts[genre_counts['genre'].isin(top_genres)]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=genre_counts_top,
        x='release_year',
        y='count',
        hue='genre',
        marker='o',
        ax=ax
    )
    ax.set_title(f'Top {top_n} Genres in the US Over Time')
    ax.set_xlabel('Release Year')
    ax.set_ylabel('Number of Projects')
    ax.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig