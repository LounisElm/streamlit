import streamlit as st
import pandas as pd

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
MOVIES_PATH = "data/hackathon/content/movie.csv"

st.title("Système de recommandation de films")
st.write("Choisissez un ensemble de recommandations puis un utilisateur.")

selected_rec = st.selectbox("Source des recommandations", list(REC_PATHS.keys()))
recs = load_recommendations(REC_PATHS[selected_rec])
try:
    movies = load_movies(MOVIES_PATH)
    id_col = "movieId" if "movieId" in movies.columns else movies.columns[0]
    title_col = "title" if "title" in movies.columns else movies.columns[1]
    id_to_title = dict(zip(movies[id_col], movies[title_col]))
except FileNotFoundError:
    st.warning("Fichier movie.csv introuvable : les titres ne seront pas affich\xC3\xA9s.")
    id_to_title = {}
user_ids = recs["user"].unique()
user_id = st.selectbox("Utilisateur", sorted(user_ids))
num_recs = st.slider("Nombre de recommandations", 1, 20, 10)

if st.button("Afficher les recommandations"):
    user_recs = recs[recs["user"] == user_id].nlargest(int(num_recs), "estimated_rating")
    st.write(f"Recommandations pour l'utilisateur {user_id} :")
    for _, row in user_recs.iterrows():
        movie_title = id_to_title.get(row["item"], f"Film {row['item']}")
        st.write(f"{movie_title} - Score prédit : {row['estimated_rating']:.2f}")

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
