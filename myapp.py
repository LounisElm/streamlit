import streamlit as st
import pandas as pd
import requests
import os
import hashlib
import random

st.set_page_config(page_title="Cinéma", layout="wide")

# Style inspiré de Netflix
st.markdown(
    """
    <style>
    .main {
        background-color: #141414;
        color: white;
    }
    .description {
        text-align: justify;
        line-height: 1.4;
    }
    .rating-container {
        display: flex;
        gap: 0.5rem;
        align-items: center;
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
def fetch_movie_details(imdb_id: int | str | None) -> dict:
    """Return additional movie details from the OMDb API."""
    if not imdb_id:
        return {}
    imdb_id = f"tt{int(imdb_id):07d}"
    url = f"https://www.omdbapi.com/?i={imdb_id}&apikey=thewdb&plot=short"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("Response") == "True":
            details = {
                "poster": data.get("Poster") if data.get("Poster") != "N/A" else "",
                "runtime": data.get("Runtime"),
                "actors": data.get("Actors"),
                "imdbRating": data.get("imdbRating"),
                "title": data.get("Title"),
                "plot": data.get("Plot") if data.get("Plot") != "N/A" else "",
            }
            return details
    except Exception:
        pass
    return {}


def fetch_trailer_url(tmdb_id: int | str | None) -> str:
    """Return a YouTube trailer URL using the TMDb API if a key is provided."""
    api_key = os.environ.get("TMDB_API_KEY")
    if not (api_key and tmdb_id):
        return ""
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/videos?api_key={api_key}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        for video in data.get("results", []):
            if video.get("site", "").lower() == "youtube":
                return f"https://www.youtube.com/watch?v={video['key']}"
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


@st.cache_data
def load_global_ratings(path: str) -> pd.Series:
    """Compute the global mean rating for each movie."""
    ratings = pd.read_csv(path)
    return ratings.groupby("movieId")["rating"].mean()


def append_rating(user: int, movie: int, rating: float) -> None:
    """Append a single rating to both user rating files."""
    row = pd.DataFrame(
        [[user, movie, rating, int(pd.Timestamp.now().timestamp())]],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    for path in [RATINGS_COPY_PATH, RATINGS_ALL_PATH]:
        header = not os.path.exists(path)
        row.to_csv(path, mode="a", header=header, index=False)
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df = df.sort_values(["userId", "timestamp"])  # keep users grouped
        else:
            df = df.sort_values(["userId"])
        df.to_csv(path, index=False)

def get_random_top_movies(n: int = 10) -> list[dict]:
    """Return ``n`` movies chosen randomly from a curated pool."""
    if not FEATURED_POOL_IDS:
        return []
    chosen = random.sample(FEATURED_POOL_IDS, k=min(n, len(FEATURED_POOL_IDS)))
    results = []
    for movie_id in chosen:
        details = fetch_movie_details(id_to_imdb.get(movie_id))
        details["movieId"] = movie_id
        details["title"] = details.get("title") or id_to_title.get(movie_id, f"Film {movie_id}")
        results.append(details)

    return results


def show_movie_details(movie_id: int, user_id: int | None, state_key: str) -> None:
    """Display movie details with rating option."""
    details = fetch_movie_details(id_to_imdb.get(movie_id))
    title = details.get("title") or id_to_title.get(movie_id, f"Film {movie_id}")

    with st.expander(title, expanded=True):
        col_main, col_side = st.columns([3, 1])

        with col_main:
            if details.get("plot"):
                st.markdown(
                    f"<div class='description'>{details['plot']}</div>",
                    unsafe_allow_html=True,
                )
            genres = ""
            if not movies.empty and "genres" in movies.columns:
                match = movies[movies[id_col] == movie_id]
                if not match.empty:
                    genres = match.iloc[0]["genres"].replace("|", ", ")
            if genres:
                st.write(f"Genres : {genres}")
            if details.get("runtime"):
                st.write(f"Durée : {details['runtime']}")
            if details.get("actors"):
                st.write(f"Acteurs : {details['actors']}")
            if movie_id in global_ratings.index:
                st.write(f"Note moyenne : {global_ratings[movie_id]:.2f}/5")
            if details.get("imdbRating"):
                st.write(f"Note IMDb : {details['imdbRating']}")

            tmdb_id = id_to_tmdb.get(movie_id)
            imdb_id_val = id_to_imdb.get(movie_id)
            trailer_url = fetch_trailer_url(tmdb_id)
            links = []
            if imdb_id_val:
                links.append(
                    f"[IMDb](https://www.imdb.com/title/tt{int(imdb_id_val):07d}/)"
                )
            if tmdb_id:
                links.append(
                    f"[TMDB](https://www.themoviedb.org/movie/{int(tmdb_id)})"
                )
            if links:
                st.markdown(" • ".join(links))
            if trailer_url:
                if st.button(
                    "Bande-annonce",
                    key=f"trailer_{state_key}_{movie_id}",
                ):
                    st.video(trailer_url)
            elif os.environ.get("TMDB_API_KEY"):
                st.info("Bande-annonce non trouvée.")
            else:
                st.info(
                    "Définissez la variable d'environnement TMDB_API_KEY pour "
                    "afficher les bandes-annonces."
                )

        with col_side:
            if details.get("poster"):
                st.image(details["poster"], use_container_width=True)
            if user_id is not None:
                slider_col, button_col = st.columns([2, 1])
                user_rating = slider_col.slider(
                    "Votre note",
                    0.0,
                    5.0,
                    2.5,
                    0.5,
                    key=f"user_rating_{state_key}_{movie_id}",
                )
                if button_col.button(
                    "Enregistrer",
                    key=f"save_rating_{state_key}_{movie_id}",
                ):
                    append_rating(user_id, movie_id, user_rating)
                    st.success("Note enregistrée")
            else:
                st.info("Sélectionnez un utilisateur pour noter ce film.")

        if st.button("Fermer", key=f"close_details_{state_key}"):
            st.session_state.pop(state_key, None)


REC_PATHS = {
    "Full dataset": "RECOMMENDER-SYSTEM/mlsmm2156/top_n_full.csv",
    "Leave-one-out": "RECOMMENDER-SYSTEM/mlsmm2156/top_n_loo.csv",
}

METRICS_PATH = "RECOMMENDER-SYSTEM/mlsmm2156/evaluation/results_all.csv"
MOVIES_PATH = "movies.csv"
LINKS_PATH = "links.csv"
RATINGS_PATH = "ratings.csv"
RATINGS_ALL_PATH = "ratings_all_users.csv"
RATINGS_COPY_PATH = "ratings_copy.csv"
PROFILE_PATH = "profiles.csv"

st.title("Système de recommandation de films")
st.write("Choisissez un ensemble de recommandations puis un utilisateur.")

# Chargement des métadonnées des films pour la recherche et l'affichage
try:
    movies = load_movies(MOVIES_PATH)
    links = load_links(LINKS_PATH)
    global_ratings = load_global_ratings(RATINGS_PATH)
    id_col = "movieId" if "movieId" in movies.columns else movies.columns[0]
    title_col = "title" if "title" in movies.columns else movies.columns[1]
    id_to_title = dict(zip(movies[id_col], movies[title_col]))
    link_id_col = "movieId" if "movieId" in links.columns else links.columns[0]
    imdb_col = "imdbId" if "imdbId" in links.columns else links.columns[1]
    tmdb_col = "tmdbId" if "tmdbId" in links.columns else links.columns[2]
    id_to_imdb = (
        dict(zip(links[link_id_col], links[imdb_col])) if not links.empty else {}
    )
    id_to_tmdb = (
        dict(zip(links[link_id_col], links[tmdb_col])) if not links.empty else {}
    )
    year_series = movies[title_col].str.extract(r"\((\d{4})\)", expand=False)
    movies["year"] = pd.to_numeric(year_series, errors="coerce")
    years = movies["year"].dropna()
    YEAR_MIN = int(years.min()) if not years.empty else 1900
    YEAR_MAX = int(years.max()) if not years.empty else 2020
    if not movies.empty and "genres" in movies.columns:
        genre_set = set()
        for g in movies["genres"].dropna():
            genre_set.update(g.split("|"))
        AVAILABLE_GENRES = sorted(genre_set)
    else:
        AVAILABLE_GENRES = []
    pool_df = movies.dropna(subset=["year"])  # keep only movies with a year
    pool_df = pool_df[pool_df["year"] >= 2005]
    rated_pool = pool_df.merge(
        global_ratings.rename("mean_rating"),
        left_on=id_col,
        right_index=True,
        how="left",
    )
    rated_pool = rated_pool.dropna(subset=["mean_rating"])
    FEATURED_POOL_IDS = (
        rated_pool.sort_values("mean_rating", ascending=False)
        .head(100)[id_col]
        .tolist()
    )
except FileNotFoundError:
    st.warning(
        f"Fichier {MOVIES_PATH} introuvable : les titres ne seront pas affichés."
    )
    movies = pd.DataFrame()
    id_to_title = {}
    id_to_imdb = {}
    id_to_tmdb = {}
    id_col = title_col = None
    global_ratings = pd.Series(dtype=float)
    FEATURED_POOL_IDS = []
    YEAR_MIN = 1900
    YEAR_MAX = 2020
    AVAILABLE_GENRES = []

# Interface principale en onglets
tab_featured, tab_rec, tab_users, tab_rated = st.tabs(
    [
        "À la une",
        "Recommandations",
        "Utilisateurs",
        "Films notés",
    ]
)

# Barre latérale pour la recherche et la sélection d'utilisateur
with st.sidebar:
    st.header("Navigation")
    selected_rec = st.selectbox("Source des recommandations", list(REC_PATHS.keys()))
    recs = load_recommendations(REC_PATHS[selected_rec])

    user_ids = sorted(recs["user"].unique())
    user_options = ["All users"] + [str(uid) for uid in user_ids]

    if "active_user_id" in st.session_state:
        active_label = f"Profil actif ({st.session_state['active_pseudo']})"
        user_options.insert(0, active_label)

    selected_user = st.selectbox("Utilisateur", user_options)

    if selected_user.startswith("Profil actif"):
        user_id = st.session_state["active_user_id"]
    else:
        user_id = None if selected_user == "All users" else int(selected_user)

    movie_query = st.text_input("Rechercher un film")

with tab_featured:
    st.subheader("Films bien notés")
    if "featured_movies" not in st.session_state:
        st.session_state["featured_movies"] = get_random_top_movies()
    if st.button("Rafraîchir", key="refresh_featured"):
        st.session_state["featured_movies"] = get_random_top_movies()
    featured_movies = st.session_state["featured_movies"]
    if not featured_movies:
        st.write("Aucun film à afficher.")
    else:
        for start in range(0, len(featured_movies), 5):
            subset = featured_movies[start : start + 5]
            cols = st.columns(len(subset))
            for col, details in zip(cols, subset):
                with col:
                    st.text(details["title"])
                    if details.get("poster"):
                        st.image(details["poster"], use_container_width=True)
                    if st.button(
                        "Description",
                        key=f"feat_{details['movieId']}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_featured"] = details["movieId"]
    if "selected_featured" in st.session_state:
        show_movie_details(
            st.session_state["selected_featured"],
            user_id,
            "selected_featured",
        )
        st.markdown("---")

with tab_rec:
    if movie_query:
        if movies.empty:
            st.info("Aucun film n'est disponible.")
        else:
            results = movies[
                movies[title_col].str.contains(movie_query, case=False, na=False)
            ]
            if results.empty:
                st.info("Aucun film trouv\u00e9.")
            else:
                for start in range(0, min(len(results), 10), 5):
                    subset = results.iloc[start : start + 5]
                    cols = st.columns(len(subset))
                    for col, (_, row) in zip(cols, subset.iterrows()):
                        with col:
                            st.text(row[title_col])
                            poster_url = (
                                fetch_poster(id_to_imdb.get(row[id_col]))
                                if id_col
                                else ""
                            )
                            if poster_url:
                                st.image(poster_url)
                            if st.button(
                                "Description",
                                key=f"search_{row[id_col]}",
                                use_container_width=True,
                            ):
                                st.session_state["selected_movie"] = row[id_col]
        st.markdown("---")

    trending_container = st.container()
    with trending_container:
        if not movies.empty and user_id is not None:
            st.subheader(f"\u00c0 la une pour l'utilisateur {user_id}")

            genre_filter = st.multiselect(
                "Filtrer par genre",
                AVAILABLE_GENRES,
                key="trend_genres",
            )
            year_range = st.slider(
                "Ann\u00e9e de sortie",
                YEAR_MIN,
                YEAR_MAX,
                (YEAR_MIN, YEAR_MAX),
                key="trend_years",
            )
            min_rating = st.slider(
                "Note minimale",
                0.0,
                5.0,
                0.0,
                0.5,
                key="trend_rating",
            )

            trending = recs[recs["user"] == user_id]
            trending = trending.merge(
                movies[[id_col, "genres", "year"]],
                left_on="item",
                right_on=id_col,
                how="left",
            )
            trending = trending[trending["estimated_rating"] >= min_rating]
            trending = trending[trending["year"].between(*year_range)]
            if genre_filter:
                trending = trending[
                    trending["genres"].apply(
                        lambda g: any(gen in g.split("|") for gen in genre_filter)
                        if isinstance(g, str)
                        else False
                    )
                ]

            trending = trending.nlargest(12, "estimated_rating")

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
                        if st.button(
                            "Description",
                            key=f"trend_{row['item']}",
                            use_container_width=True,
                        ):
                            st.session_state["selected_movie"] = row["item"]
        elif user_id is None:
            st.info("Sélectionnez un utilisateur pour voir les recommandations.")


    if "selected_movie" in st.session_state:
        show_movie_details(
            st.session_state["selected_movie"],
            user_id,
            "selected_movie",
        )
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

with tab_users:
    st.subheader("Liste des utilisateurs")
    if os.path.exists(PROFILE_PATH):
        profiles = pd.read_csv(PROFILE_PATH)
    else:
        profiles = pd.DataFrame(columns=["userId", "pseudo", "password"])
    if not profiles.empty:
        st.dataframe(profiles[["userId", "pseudo", "password"]])
    else:
        st.write("Aucun utilisateur enregistré.")

    st.markdown("---")
    st.subheader("Création d'un profil")

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
            # Determine next available user id based on existing ratings and profiles
            if os.path.exists(RATINGS_ALL_PATH):
                ratings_all = pd.read_csv(RATINGS_ALL_PATH)
                max_rating_id = (
                    ratings_all["userId"].max() if not ratings_all.empty else 0
                )
            else:
                max_rating_id = 0

            if os.path.exists(PROFILE_PATH):
                profiles = pd.read_csv(PROFILE_PATH)
                max_profile_id = profiles["userId"].max() if not profiles.empty else 0
            else:
                profiles = pd.DataFrame(columns=["userId", "pseudo", "password"])
                max_profile_id = 0

            next_id = max(max_rating_id, max_profile_id) + 1

            hashed_pw = hashlib.sha256(password.encode()).hexdigest()

            profiles = profiles._append(
                {"userId": next_id, "pseudo": pseudo, "password": hashed_pw},
                ignore_index=True,
            )
            profiles.to_csv(PROFILE_PATH, index=False)

            if os.path.exists(RATINGS_COPY_PATH):
                ratings_ext = pd.read_csv(RATINGS_COPY_PATH)
            else:
                ratings_ext = pd.read_csv("ratings.csv")

            genre_offset = 1_000_000
            genre_ids = {g: genre_offset + i for i, g in enumerate(genres_list)}
            now_ts = int(pd.Timestamp.now().timestamp())
            new_rows = []
            for g, r in genre_ratings.items():
                row = {
                    "userId": next_id,
                    "movieId": genre_ids[g],
                    "rating": r,
                    "timestamp": now_ts,
                }
                ratings_ext = ratings_ext._append(row, ignore_index=True)
                new_rows.append(row)

            ratings_ext = ratings_ext.sort_values(["userId", "timestamp"])
            ratings_ext.to_csv(RATINGS_COPY_PATH, index=False)

            if new_rows:
                if os.path.exists(RATINGS_ALL_PATH):
                    ratings_all = pd.read_csv(RATINGS_ALL_PATH)
                else:
                    ratings_all = pd.DataFrame(
                        columns=["userId", "movieId", "rating", "timestamp"]
                    )
                ratings_all = pd.concat(
                    [ratings_all, pd.DataFrame(new_rows)], ignore_index=True
                )
                ratings_all = ratings_all.sort_values(["userId", "timestamp"])
                ratings_all.to_csv(RATINGS_ALL_PATH, index=False)
            st.success(f"Profil enregistré avec l'identifiant {next_id}.")

    st.markdown("---")
    st.subheader("Connexion au profil")
    login_pseudo = st.text_input("Pseudonyme", key="login_pseudo")
    login_password = st.text_input(
        "Mot de passe", type="password", key="login_password"
    )
    if st.button("Se connecter"):
        if os.path.exists(PROFILE_PATH):
            profiles = pd.read_csv(PROFILE_PATH)
            row = profiles[profiles["pseudo"] == login_pseudo]
            if not row.empty:
                hashed = hashlib.sha256(login_password.encode()).hexdigest()
                if hashed == row.iloc[0]["password"]:
                    st.success(f"Connecté en tant que {login_pseudo}")

                    st.session_state["active_user_id"] = int(row.iloc[0]["userId"])
                    st.session_state["active_pseudo"] = login_pseudo
                else:
                    st.error("Mot de passe incorrect.")
            else:
                st.error("Pseudonyme inconnu.")
        else:
            st.error("Aucun profil enregistré.")

with tab_rated:
    st.subheader("Films d\u00e9j\u00e0 not\u00e9s")
    if os.path.exists(RATINGS_ALL_PATH):
        ratings_all = pd.read_csv(RATINGS_ALL_PATH)
    else:
        ratings_all = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

    merged = ratings_all.merge(
        movies[[id_col, title_col]], left_on="movieId", right_on=id_col, how="left"
    )
    merged = merged[["userId", "movieId", title_col, "rating"]]

    # Show only the ratings for the selected user (or all users)
    if user_id is None:
        user_ratings = merged
    else:
        user_ratings = merged[merged["userId"] == user_id]
    st.dataframe(user_ratings.head(100))
    if not user_ratings.empty:
        csv = user_ratings.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger le CSV",
            data=csv,
            file_name="user_ratings.csv",
            mime="text/csv",
        )
