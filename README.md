# Streamlit Movie Recommender

This repository contains a simple Streamlit application demonstrating a movie recommendation system.

The app lets you search for movies and view personalized recommendations. Movie posters are displayed with a **Description** button for more info.

The original version could optionally show YouTube trailers when a `TMDB_API_KEY` was provided. This feature has been removed to simplify the interface.

## User profiles

The app now supports creating simple user profiles. A new profile is assigned
an incremental identifier following the last existing user ID in the ratings
dataset. Passwords are stored using a SHA-256 hash and displayed hashed in the
"Liste des utilisateurs" table. A basic login form allows you to test profile
authentication within the app.

When a user logs in successfully, the sidebar user selector now includes a
**Profil actif** option synced with the connected user. Selecting this option
lets you rate movies and export those ratings together with the other users.
