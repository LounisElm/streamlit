import streamlit as st
import pandas as pd
import requests
import os
import uuid

st.set_page_config(page_title="Cinéma", layout="wide")

# Style inspiré de Netflix
st.markdown(
    """
    <style>
    .main {
        background-color: #141414;
        color: white;
    }
    .search-container {
        position: fixed;
        top: 10px;
        right: 10px;
        width: 200px;
        z-index: 100;
    }
    .search-container input {
        font-size: 12px;
        padding: 2px 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_poster(imdb_id: int | str | None) -> str:
    """Return the poster URL for a given IMDb id using the free OMDb API."""
    if not imdb_id:
        return ""
    imdb_id = f"tt{int(imdb_id):07d}"
    url = f"https://www.omdbapi.com/?i={imdb_id}&apikey=thewdb"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_url = data.get("Poster")
        if poster_url and poster_url != "N/A":
            return poster_url
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

@st.cache_data
def load_links(path: str) -> pd.DataFrame:
    """Load mapping between MovieLens ids and IMDb/TMDb ids."""
    return pd.read_csv(path)

REC_PATHS = {
    "Full dataset": "RECOMMENDER-SYSTEM/mlsmm2156/top_n_full.csv",
    "Leave-one-out": "RECOMMENDER-SYSTEM/mlsmm2156/top_n_loo.csv",
}

METRICS_PATH = "RECOMMENDER-SYSTEM/mlsmm2156/evaluation/results_all.csv"
MOVIES_PATH = "movies.csv"
LINKS_PATH = "links.csv"

st.title("Système de recommandation de films")
st.write("Choisissez un ensemble de recommandations puis un utilisateur.")

# Chargement des métadonnées des films pour la recherche et l'affichage
try:
    movies = load_movies(MOVIES_PATH)
    links = load_links(LINKS_PATH)
    id_col = "movieId" if "movieId" in movies.columns else movies.columns[0]
    title_col = "title" if "title" in movies.columns else movies.columns[1]
    id_to_title = dict(zip(movies[id_col], movies[title_col]))
    link_id_col = "movieId" if "movieId" in links.columns else links.columns[0]
    imdb_col = "imdbId" if "imdbId" in links.columns else links.columns[1]
    id_to_imdb = dict(zip(links[link_id_col], links[imdb_col])) if not links.empty else {}
except FileNotFoundError:
    st.warning(f"Fichier {MOVIES_PATH} introuvable : les titres ne seront pas affichés.")
    movies = pd.DataFrame()
    id_to_title = {}
    id_to_imdb = {}
    id_col = title_col = None


# Barre de recherche de films
st.markdown("<div class='search-container'>", unsafe_allow_html=True)
movie_query = st.text_input("", placeholder="Rechercher un film")
st.markdown("</div>", unsafe_allow_html=True)
if movie_query:
    if movies.empty:
        st.info("Aucun film n'est disponible.")
    else:
        results = movies[movies[title_col].str.contains(movie_query, case=False, na=False)]
        if results.empty:
            st.info("Aucun film trouvé.")
        else:
            for start in range(0, min(len(results), 10), 5):
                subset = results.iloc[start : start + 5]
                cols = st.columns(len(subset))
                for col, (_, row) in zip(cols, subset.iterrows()):
                    with col:
                        st.text(row[title_col])
                        poster_url = fetch_poster(id_to_imdb.get(row[id_col])) if id_col else ""
                        if poster_url:
                            st.image(poster_url)
    st.markdown("---")

trending_container = st.container()
with st.container():
    selected_rec = st.selectbox("Source des recommandations", list(REC_PATHS.keys()))
    recs = load_recommendations(REC_PATHS[selected_rec])

    user_ids = recs["user"].unique()
    user_id = st.selectbox("Utilisateur", sorted(user_ids))
    num_recs = st.slider("Nombre de recommandations", 1, 20, 10)
    show_recs = st.button("Afficher les recommandations")

with trending_container:
    if not movies.empty:
        st.subheader(f"\u00c0 la une pour l'utilisateur {user_id}")
        trending = recs[recs["user"] == user_id].nlargest(12, "estimated_rating")
        for start in range(0, len(trending), 4):
            subset = trending.iloc[start : start + 4]
            cols = st.columns(len(subset))
            for col, (_, row) in zip(cols, subset.iterrows()):
                movie_title = id_to_title.get(row["item"], f"Film {row['item']}")
                poster_url = fetch_poster(id_to_imdb.get(row["item"]))
                with col:
                    st.text(movie_title)
                    if poster_url:
                        st.image(poster_url, use_container_width=True)

if show_recs:
    user_recs = recs[recs["user"] == user_id].nlargest(int(num_recs), "estimated_rating")

    # Display posters in rows of up to 5 movies
    rec_list = user_recs.to_dict(orient="records")
    for start in range(0, len(rec_list), 5):
        subset = rec_list[start : start + 5]
        cols = st.columns(len(subset))
        for col, row in zip(cols, subset):
            movie_title = id_to_title.get(row["item"], f"Film {row['item']}")
            poster_url = fetch_poster(id_to_imdb.get(row["item"]))
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

st.markdown("---")
st.subheader("Création d'un profil")

PROFILE_PATH = "profiles.csv"
RATINGS_COPY_PATH = "ratings_copy.csv"

# Préparation de la liste des genres à partir des métadonnées des films
if not movies.empty and "genres" in movies.columns:
    genre_set = set()
    for g in movies["genres"].dropna():
        genre_set.update(g.split("|"))
    genres_list = sorted(genre_set)
else:
    genres_list = []

pseudo = st.text_input("Pseudonyme")
password = st.text_input("Mot de passe", type="password")
selected_genres = st.multiselect("Genres préférés", genres_list)

genre_ratings = {}
for genre in selected_genres:
    genre_ratings[genre] = st.slider(
        f"Note pour {genre}", 0.0, 5.0, 2.5, 0.5, key=f"rating_{genre}"
    )

if st.button("Enregistrer le profil"):
    if not pseudo or not password:
        st.error("Veuillez renseigner un pseudo et un mot de passe.")
    else:
        # Chargement ou création du fichier de profils
        if os.path.exists(PROFILE_PATH):
            profiles = pd.read_csv(PROFILE_PATH)
            next_id = profiles["userId"].max() + 1 if not profiles.empty else 1
        else:
            profiles = pd.DataFrame(columns=["userId", "pseudo", "password"])
            next_id = 1

        profiles = profiles._append(
            {"userId": next_id, "pseudo": pseudo, "password": password},
            ignore_index=True,
        )
        profiles.to_csv(PROFILE_PATH, index=False)

        # Chargement ou création du fichier ratings étendu
        if os.path.exists(RATINGS_COPY_PATH):
            ratings_ext = pd.read_csv(RATINGS_COPY_PATH)
        else:
            ratings_ext = pd.read_csv("ratings.csv")

        genre_offset = 1_000_000
        genre_ids = {g: genre_offset + i for i, g in enumerate(genres_list)}
        now_ts = int(pd.Timestamp.now().timestamp())
        for g, r in genre_ratings.items():
            ratings_ext = ratings_ext._append(
                {
                    "userId": next_id,
                    "movieId": genre_ids[g],
                    "rating": r,
                    "timestamp": now_ts,
                },
                ignore_index=True,
            )

        ratings_ext.to_csv(RATINGS_COPY_PATH, index=False)
        st.success(f"Profil enregistré avec l'identifiant {next_id}.")
