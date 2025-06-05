import streamlit as st

st.title("Système de recommandation de films")
st.write("Bienvenue sur l’interface de recommandation de films.")

# Simule un utilisateur
user_id = st.number_input("Entrez votre identifiant utilisateur", min_value=1, value=1)

# Bouton pour lancer la recommandation
if st.button("Obtenir des recommandations"):
    # Ici tu mettras l'appel à ton vrai modèle plus tard
    st.write(f"Voici les recommandations pour l'utilisateur {user_id} :")
    # Recommandations factices pour l’instant
    fake_movies = ["Inception", "Le Seigneur des Anneaux", "Matrix"]
    for movie in fake_movies:
        st.write(f"- {movie}")
genre = st.selectbox("Choisissez un genre", ["Tous", "Action", "Comédie", "Drame"])
