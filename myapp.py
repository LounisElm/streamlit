import streamlit as st
import pandas as pd
import requests


def fetch_poster(movie_id: int) -> str:
    """Return the poster URL for a given movie id using TMDb API."""
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}?"
        "api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
    )
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        pass
    return ""

@st.cache_data
def load_recommendations(path: str) -> pd.DataFrame:
    """Load pre-computed top-n recommendations."""
    return pd.read_csv(path)

@st.cache_data
def load_movies(path: str) -> pd.DataFrame:
    """Load movie metadata for title lookup."""
    return pd.read_csv(path)

@st.cache_data
def load_metrics(path: str) -> pd.DataFrame:
    """Load offline evaluation metrics."""
    return pd.read_csv(path)

REC_PATHS = {
    "Full dataset": "RECOMMENDER-SYSTEM/mlsmm2156/top_n_full.csv",
    "Leave-one-out": "RECOMMENDER-SYSTEM/mlsmm2156/top_n_loo.csv",
}

METRICS_PATH = "RECOMMENDER-SYSTEM/mlsmm2156/evaluation/results_all.csv"
MOVIES_PATH = "movies.csv"

st.title("Système de recommandation de films")
st.write("Choisissez un ensemble de recommandations puis un utilisateur.")

selected_rec = st.selectbox("Source des recommandations", list(REC_PATHS.keys()))
recs = load_recommendations(REC_PATHS[selected_rec])

# Bloc pour charger le mapping ID -> Titre
try:
    movies = load_movies(MOVIES_PATH)
    id_col = "movieId" if "movieId" in movies.columns else movies.columns[0]
    title_col = "title" if "title" in movies.columns else movies.columns[1]
    id_to_title = dict(zip(movies[id_col], movies[title_col]))
except FileNotFoundError:
    st.warning(f"Fichier {MOVIES_PATH} introuvable : les titres ne seront pas affichés.")
    id_to_title = {}

user_ids = recs["user"].unique()
user_id = st.selectbox("Utilisateur", sorted(user_ids))
num_recs = st.slider("Nombre de recommandations", 1, 20, 10)

if st.button("Afficher les recommandations"):
    user_recs = recs[recs["user"] == user_id].nlargest(int(num_recs), "estimated_rating")
    st.write(f"Recommandations pour l'utilisateur {user_id} :")

    # Display posters in rows of up to 5 movies
    rec_list = user_recs.to_dict(orient="records")
    for start in range(0, len(rec_list), 5):
        subset = rec_list[start : start + 5]
        cols = st.columns(len(subset))
        for col, row in zip(cols, subset):
            movie_title = id_to_title.get(row["item"], f"Film {row['item']}")
            poster_url = fetch_poster(row["item"])
            with col:
                st.text(movie_title)
                if poster_url:
                    st.image(poster_url)
                st.caption(f"Score prédit : {row['estimated_rating']:.2f}")

st.markdown("---")

st.subheader("Performances des modèles (hors ligne)")
metrics_df = load_metrics(METRICS_PATH)
st.dataframe(metrics_df)

st.markdown(
    """
### À propos de ce système
Ce démonstrateur combine différentes approches de recommandation :
- **Collaboratif utilisateur** avec une métrique de similarité personnalisée ;
- **Filtrage par contenu** utilisant les caractéristiques des films ;
- **Modèle à facteurs latents** pour capturer les préférences implicites.

Les meilleures prédictions issues de ces modèles ont été pré‑calculées et
sont chargées dans cette application. Le tableau ci-dessus résume les
performances obtenues lors de l'évaluation.
"""
)
