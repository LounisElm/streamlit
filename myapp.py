import streamlit as st
import pandas as pd

@st.cache_data
def load_recommendations(path: str) -> pd.DataFrame:
    """Load the pre-computed recommendations."""
    return pd.read_csv(path)

# Load data
REC_PATH = "RECOMMENDER-SYSTEM/mlsmm2156/top_n_full.csv"
recs = load_recommendations(REC_PATH)

st.title("Système de recommandation de films")
st.write("Sélectionnez un utilisateur pour afficher ses recommandations.")

user_ids = recs['user'].unique()
user_id = st.selectbox("Utilisateur", sorted(user_ids))
num_recs = st.number_input("Nombre de recommandations", min_value=1, max_value=20, value=10)

if st.button("Afficher"):
    user_recs = recs[recs['user'] == user_id].nlargest(int(num_recs), 'estimated_rating')
    st.write(f"Recommandations pour l'utilisateur {user_id} :")
    for _, row in user_recs.iterrows():
        st.write(f"Film {row['item']} - Score prédit : {row['estimated_rating']:.2f}")
