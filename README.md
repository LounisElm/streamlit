# Streamlit Movie Recommender

This repository contains a simple Streamlit application demonstrating a movie recommendation system.

The app lets you search for movies and view personalized recommendations. Movie posters are displayed with a **Description** button for more info.

The **À la une** section now keeps the same selection while you browse. Use the new **Rafraîchir** button to see another set of movies.

If a `TMDB_API_KEY` environment variable is defined, the app now fetches a
YouTube trailer for each movie when available. Links to the corresponding IMDb
and TMDB pages are also displayed in the description panel.

The recommendation tab includes filters for genre, release year and minimum
predicted rating so you can refine the suggestions.

## User profiles

The app now supports creating simple user profiles. A new profile is assigned
an incremental identifier following the last existing user ID in the ratings
dataset. Passwords are stored using a SHA-256 hash and displayed hashed in the
"Liste des utilisateurs" table. A basic login form allows you to test profile
authentication within the app.

When creating a profile you are asked to rate at least 10 movies. A dedicated
"Créer un profil" tab presents ten randomly selected well-rated titles (a refresh
button lets you change the selection while keeping the ratings already given).
Once ten movies are rated, you can submit the profile and a quick user-based
algorithm suggests ten movies you might enjoy. A rating of ``0`` simply means
that no rating was provided for that title.


When a user logs in successfully, the sidebar user selector now includes a
**Profil actif** option synced with the connected user. Selecting this option
lets you rate movies and export those ratings together with the other users.

## Installation

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file lists `streamlit`, `pandas`, `scikit-learn` and the
other dependencies needed to run the application.
