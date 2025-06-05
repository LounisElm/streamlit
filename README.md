# Streamlit Movie Recommender

This repository contains a simple Streamlit application demonstrating a movie recommendation system.

The app lets you search for movies and view personalized recommendations. Movie posters are displayed with a **Description** button for more info.

## Trailer display

If you set the environment variable `TMDB_API_KEY` with your TMDb API key, the
application will retrieve the YouTube trailer for each movie when available and
show it alongside the plot description using `st.video` in the details panel.
