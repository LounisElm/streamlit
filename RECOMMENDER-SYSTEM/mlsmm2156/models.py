# standard library imports
from collections import defaultdict
import random
from loaders import load_items, load_ratings, load_genome_data
from constants import Constant as C

# third parties imports
import numpy as np
import random as rd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from surprise.prediction_algorithms.predictions import PredictionImpossible
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import time
import matplotlib.pyplot as plt # Pour la visualisation (optionnel pour l'exécution directe)
from sklearn.impute import KNNImputer
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

def genre_tokenizer(x):
    return x.split('|') if pd.notna(x) else []


def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
    Source: inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    and modified by cvandekerckh for random tie breaking

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    rd.seed(0)

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        rd.shuffle(user_ratings)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First algorithm
class ModelBaseline1(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        return 2


# Second algorithm
class ModelBaseline2(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        rd.seed(0)

    def estimate(self, u, i):
        return rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])


# Third algorithm
class ModelBaseline3(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        return self

    def estimate(self, u, i):
        return self.the_mean


# Fourth Model
class ModelBaseline4(SVD):
    def __init__(self):
        SVD.__init__(self, n_factors=100,random_state=1)
        
        
# class ContentBased(AlgoBase):
#     """
#     Content-based recommender system that uses movie features to predict user preferences.
    
#     Features:
#     - Basic features (title length, year, age)
#     - Textual features (TF-IDF on genres)
#     - Statistical features (genre popularity, user stats)
#     - Additional features (genome tags, visual features)
    
#     Regressors:
#     - Linear Regression
#     - Random Forest
#     - XGBoost
#     - Neural Network
#     """
    
#     def __init__(self, features_method, regressor_method, nn_config=None):
#         """
#         Initialize the content-based recommender.
        
#         Args:
#             features_method (str): Method to extract features
#             regressor_method (str): Regressor to use
#             nn_config (dict, optional): Neural network configuration
#         """
#         AlgoBase.__init__(self)
#         self.regressor_method = regressor_method
#         self.features_method = features_method
#         self.nn_config = nn_config
#         self.user_profile_explain = {}  # Dictionary to store explanations
        
#         # Initialize tfidf_vectorizer BEFORE calling create_content_features
#         self.tfidf_vectorizer = TfidfVectorizer(tokenizer=genre_tokenizer)

        
#         # Initialize device for neural network
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if self.device.type == 'cuda':
#             torch.cuda.empty_cache()
#             print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
#         # Training configuration
#         self.training_config = {
#             'batch_size': 32,
#             'epochs': 50,
#             'early_stopping_patience': 5,
#             'learning_rate': 0.001
#         }
        
#         # Initialize features
#         self.content_features = self.create_content_features(features_method)
        
#         if regressor_method == "neural_network":
#             self.model = None  # Will be initialized in fit()
#             self.optimizer = None
#             self.criterion = nn.MSELoss()

#     def create_content_features(self, features_method):
#         """Content Analyzer"""
#         df_items = load_items()
#         df_ratings = load_ratings()
        
#         if features_method is None:
#             df_features = None
#         elif features_method == "title_length":
#             df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')
#         elif features_method == "title_year":
#             df_features = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)', expand=False).to_frame('title_year')
#             df_features['title_year'] = pd.to_numeric(df_features['title_year'], errors='coerce')
#         elif features_method == "movie_age":
#             df_features = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)', expand=False).to_frame('title_year')
#             df_features['title_year'] = pd.to_numeric(df_features['title_year'], errors='coerce')
#             current_year = pd.Timestamp.now().year
#             df_features['movie_age'] = current_year - df_features['title_year']
#         elif features_method == "decade":
#             df_features = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)', expand=False).to_frame('title_year')
#             df_features['title_year'] = pd.to_numeric(df_features['title_year'], errors='coerce')
#             df_features['decade'] = (df_features['title_year'] // 10) * 10
#             df_features = pd.get_dummies(df_features['decade'], prefix='decade')
#         elif features_method == "genre_popularity":
#             genre_ratings = []
#             for _, row in df_items.iterrows():
#                 if pd.notna(row[C.GENRES_COL]) and row[C.GENRES_COL] != '(no genres listed)':
#                     genres = row[C.GENRES_COL].split('|')
#                     movie_ratings = df_ratings[df_ratings['movieId'] == row.name]['rating']
#                     if not movie_ratings.empty:
#                         for genre in genres:
#                             genre_ratings.append({'genre': genre, 'rating': movie_ratings.mean()})
            
#             genre_ratings_df = pd.DataFrame(genre_ratings)
#             genre_means = genre_ratings_df.groupby('genre')['rating'].mean()
            
#             df_features = pd.DataFrame(index=df_items.index)
#             for genre in genre_means.index:
#                 df_features[f'genre_{genre}_popularity'] = df_items[C.GENRES_COL].apply(
#                     lambda x: genre_means[genre] if pd.notna(x) and genre in x.split('|') else 0
#                 )
#         elif features_method == "user_stats":
#             user_stats = df_ratings.groupby('userId').agg({
#                 'rating': ['mean', 'std', 'count']
#             }).reset_index()
            
#             user_stats.columns = ['userId', 'user_mean_rating', 'user_rating_std', 'user_rating_count']
            
#             global_mean = df_ratings['rating'].mean()
#             user_stats['user_bias'] = user_stats['user_mean_rating'] - global_mean
            
#             movie_user_stats = []
#             for movie_id in df_items.index:
#                 movie_ratings = df_ratings[df_ratings['movieId'] == movie_id]
#                 if not movie_ratings.empty:
#                     movie_user_stats.append({
#                         'movieId': movie_id,
#                         'avg_user_mean': movie_ratings.merge(user_stats, on='userId')['user_mean_rating'].mean(),
#                         'avg_user_std': movie_ratings.merge(user_stats, on='userId')['user_rating_std'].mean(),
#                         'avg_user_bias': movie_ratings.merge(user_stats, on='userId')['user_bias'].mean()
#                     })
            
#             df_features = pd.DataFrame(movie_user_stats).set_index('movieId')
#         elif features_method == "all_features":
#             features_list = []
            
#             # Title length
#             features_list.append(df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title'))
            
#             # Movie age
#             year_df = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)', expand=False).to_frame('title_year')
#             year_df['title_year'] = pd.to_numeric(year_df['title_year'], errors='coerce')
#             current_year = pd.Timestamp.now().year
#             year_df['movie_age'] = current_year - year_df['title_year']
#             features_list.append(year_df)
            
#             # Decade
#             decade_df = pd.get_dummies((year_df['title_year'] // 10) * 10, prefix='decade')
#             features_list.append(decade_df)
            
#             # Genre popularity
#             genre_ratings = []
#             for _, row in df_items.iterrows():
#                 if pd.notna(row[C.GENRES_COL]) and row[C.GENRES_COL] != '(no genres listed)':
#                     genres = row[C.GENRES_COL].split('|')
#                     movie_ratings = df_ratings[df_ratings['movieId'] == row.name]['rating']
#                     if not movie_ratings.empty:
#                         for genre in genres:
#                             genre_ratings.append({'genre': genre, 'rating': movie_ratings.mean()})
            
#             genre_ratings_df = pd.DataFrame(genre_ratings)
#             genre_means = genre_ratings_df.groupby('genre')['rating'].mean()
            
#             genre_pop_df = pd.DataFrame(index=df_items.index)
#             for genre in genre_means.index:
#                 genre_pop_df[f'genre_{genre}_popularity'] = df_items[C.GENRES_COL].apply(
#                     lambda x: genre_means[genre] if pd.notna(x) and genre in x.split('|') else 0
#                 )
#             features_list.append(genre_pop_df)
            
#             # User statistics
#             user_stats = df_ratings.groupby('userId').agg({
#                 'rating': ['mean', 'std', 'count']
#             }).reset_index()
            
#             user_stats.columns = ['userId', 'user_mean_rating', 'user_rating_std', 'user_rating_count']
#             global_mean = df_ratings['rating'].mean()
#             user_stats['user_bias'] = user_stats['user_mean_rating'] - global_mean
            
#             movie_user_stats = []
#             for movie_id in df_items.index:
#                 movie_ratings = df_ratings[df_ratings['movieId'] == movie_id]
#                 if not movie_ratings.empty:
#                     movie_user_stats.append({
#                         'movieId': movie_id,
#                         'avg_user_mean': movie_ratings.merge(user_stats, on='userId')['user_mean_rating'].mean(),
#                         'avg_user_std': movie_ratings.merge(user_stats, on='userId')['user_rating_std'].mean(),
#                         'avg_user_bias': movie_ratings.merge(user_stats, on='userId')['user_bias'].mean()
#                     })
            
#             user_stats_df = pd.DataFrame(movie_user_stats).set_index('movieId')
#             features_list.append(user_stats_df)
            
#             # TF-IDF sur les genres
#             genres_processed = df_items[C.GENRES_COL].fillna('').replace('(no genres listed)', '')
#             tfidf_matrix = self.tfidf_vectorizer.fit_transform(genres_processed)
#             feature_names = self.tfidf_vectorizer.get_feature_names_out()
#             tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items.index, columns=[f"tfidf_{name}" for name in feature_names])
#             features_list.append(tfidf_df)
            
#             # Concaténer toutes les features
#             df_features = pd.concat(features_list, axis=1)
#         else:
#             raise NotImplementedError(f'Feature method {features_method} not yet implemented')
#         return df_features

#     def fit(self, trainset):
#         """Profile Learner"""
#         AlgoBase.fit(self, trainset)
        
#         # Preallocate user profiles
#         self.user_profile = {u: None for u in trainset.all_users()}
#         self.user_profile_explain = {u: {} for u in trainset.all_users()}

#         if self.content_features is None or self.content_features.empty:
#             return

#         feature_names = self.content_features.columns.tolist()
#         if not feature_names:
#             return

#         for u_inner_id in self.trainset.all_users():
#             user_ratings_data = self.trainset.ur[u_inner_id]
#             if not user_ratings_data:
#                 continue

#             # Préparer les données pour l'utilisateur
#             df_user_ratings = pd.DataFrame(user_ratings_data, columns=['inner_item_id', 'user_ratings'])
#             df_user_ratings['item_id'] = df_user_ratings['inner_item_id'].apply(lambda x: self.trainset.to_raw_iid(x))

#             df_merged = df_user_ratings.merge(
#                 self.content_features,
#                 how='left',
#                 left_on='item_id',
#                 right_index=True
#             )
            
#             df_merged_cleaned = df_merged.dropna(subset=feature_names + ['user_ratings'])

#             if df_merged_cleaned.empty or len(df_merged_cleaned) < 2:
#                 continue

#             X_user = df_merged_cleaned[feature_names].values
#             y_user = df_merged_cleaned['user_ratings'].values

#             # Calculer les explications basées sur les notes pondérées
#             total_rating = sum(y_user)
#             if total_rating > 0:
#                 for feature in feature_names:
#                     weighted_sum = 0
#                     for idx, rating in enumerate(y_user):
#                         weighted_sum += rating * X_user[idx, feature_names.index(feature)]
#                     self.user_profile_explain[u_inner_id][feature] = weighted_sum / total_rating

#             # Entraîner le modèle selon la méthode choisie
#             if self.regressor_method == 'random_score':
#                 self.user_profile[u_inner_id] = None

#             elif self.regressor_method == 'random_sample':
#                 self.user_profile[u_inner_id] = ([rating for _, rating in user_ratings_data], None)

#             elif self.regressor_method in ['linear_regression', 'random_forest', 'xgboost']:
#                 scaler = StandardScaler()
#                 try:
#                     X_user_scaled = scaler.fit_transform(X_user)
#                 except ValueError:
#                     self.user_profile[u_inner_id] = None
#                     continue

#                 try:
#                     if self.regressor_method == 'linear_regression':
#                         model = LinearRegression(fit_intercept=True)
#                         model.fit(X_user_scaled, y_user)
#                         # Pour la régression linéaire, utiliser les coefficients comme explications
#                         for feature, coef in zip(feature_names, model.coef_):
#                             self.user_profile_explain[u_inner_id][feature] = abs(coef)
#                     elif self.regressor_method == 'random_forest':
#                         model = RandomForestRegressor(
#                             n_estimators=100, 
#                             random_state=0, 
#                             n_jobs=-1, 
#                             max_features='sqrt'
#                         )
#                         model.fit(X_user_scaled, y_user)
#                         # Pour RandomForest, utiliser les importances des features
#                         for feature, importance in zip(feature_names, model.feature_importances_):
#                             self.user_profile_explain[u_inner_id][feature] = importance
#                     elif self.regressor_method == 'xgboost':
#                         model = xgb.XGBRegressor(
#                             objective='reg:squarederror',
#                             n_estimators=100,
#                             learning_rate=0.1,
#                             max_depth=5,
#                             random_state=0,
#                             n_jobs=-1
#                         )
#                         model.fit(X_user_scaled, y_user)
#                         # Pour XGBoost, utiliser les importances des features
#                         for feature, importance in zip(feature_names, model.feature_importances_):
#                             self.user_profile_explain[u_inner_id][feature] = importance

#                     self.user_profile[u_inner_id] = (model, scaler)
#                 except Exception as e:
#                     print(f"Error fitting model for user {u_inner_id}: {str(e)}")
#                     self.user_profile[u_inner_id] = None

#             elif self.regressor_method == "neural_network":
#                 # Normaliser les features
#                 scaler = StandardScaler()
#                 X_user_scaled = scaler.fit_transform(X_user)
                
#                 # Convertir en tenseurs PyTorch
#                 X_tensor = torch.FloatTensor(X_user_scaled).to(self.device)
#                 y_tensor = torch.FloatTensor(y_user).reshape(-1, 1).to(self.device)
                
#                 # Initialiser le modèle
#                 model = MovieRecommenderNN(len(feature_names)).to(self.device)
#                 optimizer = optim.Adam(model.parameters(), lr=0.001)
                
#                 # Entraînement
#                 model.train()
#                 for epoch in range(100):
#                     optimizer.zero_grad()
#                     outputs = model(X_tensor)
#                     loss = self.criterion(outputs, y_tensor)
#                     loss.backward()
#                     optimizer.step()
                
#                 # Pour le réseau de neurones, utiliser les poids de la première couche comme explications
#                 first_layer_weights = model.network[0].weight.data.abs().mean(dim=0)
#                 for feature, weight in zip(feature_names, first_layer_weights):
#                     self.user_profile_explain[u_inner_id][feature] = weight.item()
                
#                 self.user_profile[u_inner_id] = (model, scaler)
#             else:
#                 raise NotImplementedError(f"Regressor method {self.regressor_method} not implemented in fit.")

#     def estimate(self, u, i):
#         """Scoring component used for item filtering"""
#         if not self.trainset.knows_user(u): 
#              raise PredictionImpossible(f'User with inner ID {u} is unkown.')
        
#         if not self.trainset.knows_item(i):
#             raise PredictionImpossible(f'Item with inner ID {i} is unkown.')

#         profile_data = self.user_profile.get(u)

#         if self.regressor_method == 'random_score':
#             score = rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])

#         elif self.regressor_method == 'random_sample':
#             if profile_data and profile_data[0] and len(profile_data[0]) > 0:
#                 score = rd.choice(profile_data[0])
#             else: 
#                  raise PredictionImpossible(f'No ratings available to sample for user {self.trainset.to_raw_uid(u)}')
        
#         elif self.regressor_method in ['linear_regression', 'random_forest', 'xgboost']:
#             if profile_data is None:
#                 raise PredictionImpossible(f'No model/scaler for user {self.trainset.to_raw_uid(u)}.')
            
#             model, scaler = profile_data
#             if model is None or scaler is None:
#                  raise PredictionImpossible(f'Model or scaler is None for user {self.trainset.to_raw_uid(u)}.')

#             raw_item_id = self.trainset.to_raw_iid(i)

#             if self.content_features is None or raw_item_id not in self.content_features.index:
#                  raise PredictionImpossible(f'Item {raw_item_id} not in content_features or content_features is None.')

#             feature_names = self.content_features.columns.tolist()
#             if not feature_names:
#                 raise PredictionImpossible('No feature names found in content_features during estimation.')

#             item_features_series = self.content_features.loc[raw_item_id, feature_names]
            
#             if item_features_series.isna().any():
#                 raise PredictionImpossible(f'One or more features are NaN for item {raw_item_id}.')

#             X_item = item_features_series.values.reshape(1, -1)
            
#             try:
#                 X_item_scaled = scaler.transform(X_item)
#                 score = model.predict(X_item_scaled)[0]
#                 min_rating, max_rating = self.trainset.rating_scale
#                 score = np.clip(score, min_rating, max_rating)
#             except Exception as e:
#                 raise PredictionImpossible(f'Error predicting with {self.regressor_method} for user {self.trainset.to_raw_uid(u)}, item {raw_item_id}: {e}')

#         elif self.regressor_method == "neural_network":
#             if profile_data is None:
#                 raise PredictionImpossible(f'No model/scaler for user {self.trainset.to_raw_uid(u)}.')
            
#             model, scaler = profile_data
#             if model is None or scaler is None:
#                 raise PredictionImpossible(f'Model or scaler is None for user {self.trainset.to_raw_uid(u)}.')

#             raw_item_id = self.trainset.to_raw_iid(i)
            
#             if self.content_features is None or raw_item_id not in self.content_features.index:
#                 raise PredictionImpossible(f'Item {raw_item_id} not in content_features or content_features is None.')

#             feature_names = self.content_features.columns.tolist()
#             item_features_series = self.content_features.loc[raw_item_id, feature_names]
            
#             if item_features_series.isna().any():
#                 raise PredictionImpossible(f'One or more features are NaN for item {raw_item_id}.')

#             X_item = item_features_series.values.reshape(1, -1)
#             X_item_scaled = scaler.transform(X_item)
            
#             # Convertir en tenseur et faire la prédiction
#             X_tensor = torch.FloatTensor(X_item_scaled).to(self.device)
#             model.eval()
#             with torch.no_grad():
#                 score = model(X_tensor).item()
            
#             # Clipper le score dans la plage de notes valide
#             min_rating, max_rating = self.trainset.rating_scale
#             score = np.clip(score, min_rating, max_rating)
            
#             return score
#         else:
#             raise NotImplementedError(f"Regressor method {self.regressor_method} not implemented in estimate.")
            
#         return score

#     def explain(self, u):
#         """
#         Retourne l'importance de chaque feature pour l'utilisateur u.
#         Les scores sont normalisés entre 0 et 1.
        
#         Args:
#             u: ID de l'utilisateur
            
#         Returns:
#             dict: Dictionnaire {feature_name: feature_score} où feature_score ∈ [0, 1]
#         """
#         if not self.trainset.knows_user(u):
#             raise PredictionImpossible('User is unknown.')
        
#         if u not in self.user_profile_explain:
#             return {}
            
#         # Récupérer les explications brutes
#         raw_explanations = self.user_profile_explain[u]
        
#         # Normaliser les scores entre 0 et 1
#         if raw_explanations:
#             max_score = max(raw_explanations.values())
#             min_score = min(raw_explanations.values())
#             if max_score != min_score:  # Éviter la division par zéro
#                 normalized_explanations = {
#                     feature: (score - min_score) / (max_score - min_score)
#                     for feature, score in raw_explanations.items()
#                 }
#             else:
#                 normalized_explanations = {feature: 0.5 for feature in raw_explanations}
#         else:
#             normalized_explanations = {}
            
#         return normalized_explanations

#     def visualize_user_profile(self, user_id, top_n=10, save_path=None):
#         """
#         Visualise le profil d'un utilisateur en montrant les features les plus importantes.
        
#         Args:
#             user_id: ID de l'utilisateur
#             top_n: Nombre de features à afficher
#             save_path: Chemin pour sauvegarder le graphique (optionnel)
#         """
#         if not self.trainset.knows_user(user_id):
#             raise PredictionImpossible('User is unknown.')
        
#         explanations = self.explain(user_id)
#         if not explanations:
#             print(f"Aucune explication disponible pour l'utilisateur {user_id}")
#             return
        
#         # Trier les features par importance
#         sorted_features = sorted(explanations.items(), key=lambda x: x[1], reverse=True)[:top_n]
#         features, scores = zip(*sorted_features)
        
#         # Créer le graphique
#         plt.figure(figsize=(12, 6))
#         bars = plt.barh(range(len(features)), scores)
#         plt.yticks(range(len(features)), features)
#         plt.xlabel('Importance de la feature')
#         plt.title(f'Profil utilisateur {user_id} - Top {top_n} features')
        
#         # Ajouter les valeurs sur les barres
#         for bar in bars:
#             width = bar.get_width()
#             plt.text(width, bar.get_y() + bar.get_height()/2, 
#                     f'{width:.3f}', ha='left', va='center')
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path)
#             plt.close()
#         else:
#             plt.show()

#     def visualize_feature_importance_comparison(self, user_ids, top_n=10, save_path=None):
#         """
#         Compare l'importance des features entre plusieurs utilisateurs.
        
#         Args:
#             user_ids: Liste d'IDs d'utilisateurs
#             top_n: Nombre de features à afficher
#             save_path: Chemin pour sauvegarder le graphique (optionnel)
#         """
#         # Vérifier que tous les utilisateurs existent
#         for user_id in user_ids:
#             if not self.trainset.knows_user(user_id):
#                 raise PredictionImpossible(f'User {user_id} is unknown.')
        
#         # Obtenir les explications pour chaque utilisateur
#         user_explanations = {}
#         for user_id in user_ids:
#             explanations = self.explain(user_id)
#             if explanations:
#                 # Trier et garder les top_n features
#                 sorted_features = sorted(explanations.items(), key=lambda x: x[1], reverse=True)[:top_n]
#                 user_explanations[user_id] = dict(sorted_features)
        
#         if not user_explanations:
#             print("Aucune explication disponible pour les utilisateurs sélectionnés")
#             return
        
#         # Créer le graphique
#         plt.figure(figsize=(15, 8))
        
#         # Préparer les données pour le graphique
#         features = list(next(iter(user_explanations.values())).keys())
#         x = np.arange(len(features))
#         width = 0.8 / len(user_ids)
        
#         # Tracer les barres pour chaque utilisateur
#         for i, (user_id, explanations) in enumerate(user_explanations.items()):
#             scores = [explanations.get(f, 0) for f in features]
#             plt.bar(x + i*width, scores, width, label=f'User {user_id}')
        
#         plt.xlabel('Features')
#         plt.ylabel('Importance')
#         plt.title('Comparaison des profils utilisateurs')
#         plt.xticks(x + width*(len(user_ids)-1)/2, features, rotation=45, ha='right')
#         plt.legend()
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path)
#             plt.close()
#         else:
#             plt.show()

#     def visualize_feature_correlation(self, user_id, save_path=None):
#         """
#         Visualise la corrélation entre les features pour un utilisateur.
        
#         Args:
#             user_id: ID de l'utilisateur
#             save_path: Chemin pour sauvegarder le graphique (optionnel)
#         """
#         if not self.trainset.knows_user(user_id):
#             raise PredictionImpossible('User is unknown.')
        
#         # Obtenir les données de l'utilisateur
#         user_ratings = self.trainset.ur[user_id]
#         if not user_ratings:
#             print(f"Aucune donnée disponible pour l'utilisateur {user_id}")
#             return
        
#         # Créer un DataFrame avec les features et les notes
#         df_user = pd.DataFrame(user_ratings, columns=['inner_item_id', 'rating'])
#         df_user['item_id'] = df_user['inner_item_id'].apply(lambda x: self.trainset.to_raw_iid(x))
        
#         # Fusionner avec les features
#         df_merged = df_user.merge(
#             self.content_features,
#             how='left',
#             left_on='item_id',
#             right_index=True
#         )
        
#         # Calculer la matrice de corrélation
#         corr_matrix = df_merged.corr()
        
#         # Créer le graphique
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
#         plt.title(f'Matrice de corrélation des features pour l\'utilisateur {user_id}')
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path)
#             plt.close()
#         else:
#             plt.show()

#     def get_global_feature_importance(self, top_n=10):
#         """
#         Calcule et retourne les features les plus importantes globalement
        
#         Args:
#             top_n (int): Nombre de features à retourner
            
#         Returns:
#             dict: Dictionnaire des features les plus importantes avec leurs scores moyens
#         """
#         if not self.user_profile_explain:
#             return {}
            
#         # Agréger les importances des features pour tous les utilisateurs
#         feature_importances = defaultdict(list)
#         for user_explanations in self.user_profile_explain.values():
#             for feature, importance in user_explanations.items():
#                 feature_importances[feature].append(importance)
        
#         # Calculer la moyenne des importances pour chaque feature
#         global_importances = {
#             feature: np.mean(importances) 
#             for feature, importances in feature_importances.items()
#         }
        
#         # Trier les features par importance
#         sorted_features = sorted(
#             global_importances.items(), 
#             key=lambda x: x[1], 
#             reverse=True
#         )
        
#         return dict(sorted_features[:top_n])

#     def visualize_global_feature_importance(self, top_n=10, save_path=None):
#         """
#         Visualise l'importance globale des features.
        
#         Args:
#             top_n: Nombre de features à afficher
#             save_path: Chemin pour sauvegarder le graphique (optionnel)
#         """
#         global_importances = self.get_global_feature_importance(top_n)
#         if not global_importances:
#             print("Aucune donnée d'importance globale disponible")
#             return
        
#         features, scores = zip(*global_importances.items())
        
#         plt.figure(figsize=(12, 6))
#         bars = plt.barh(range(len(features)), scores)
#         plt.yticks(range(len(features)), features)
#         plt.xlabel('Importance moyenne')
#         plt.title(f'Importance globale des features - Top {top_n}')
        
#         # Ajouter les valeurs sur les barres
#         for bar in bars:
#             width = bar.get_width()
#             plt.text(width, bar.get_y() + bar.get_height()/2, 
#                     f'{width:.3f}', ha='left', va='center')
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path)
#             plt.close()
#         else:
#             plt.show()

def save_features_to_csv(features_method="all_features_with_genome", output_path="data/features"):
    """
    Sauvegarde les features générées dans un fichier CSV
    
    Args:
        features_method (str): Méthode de génération des features
        output_path (str): Chemin où sauvegarder le fichier CSV
    """
    # Créer une instance de ContentBased
    model = ContentBased(features_method=features_method, regressor_method="random_forest")
    
    # Générer les features
    features_df = model.create_content_features(features_method)
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_path, exist_ok=True)
    
    # Générer le nom du fichier avec la date et l'heure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"features_{features_method}_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    # Sauvegarder les features
    features_df.to_csv(filepath)
    print(f"Features sauvegardées dans : {filepath}")
    
    # Afficher quelques statistiques sur les features
    print("\nStatistiques sur les features :")
    print(f"Nombre de features : {features_df.shape[1]}")
    print(f"Nombre de films : {features_df.shape[0]}")
    print("\nAperçu des features :")
    print(features_df.head())
    
    return features_df

class MovieRecommenderNN(nn.Module):
    def __init__(self, input_size):
        super(MovieRecommenderNN, self).__init__()
        
        # Architecture du réseau
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)


# # Créer et entraîner le modèle
# sp_ratings = load_ratings(surprise_format=True)
# trainset = sp_ratings.build_full_trainset()
# model = ContentBased(features_method="all_features", regressor_method="random_forest")
# model.fit(trainset)

# # Sélectionner un utilisateur aléatoire
# random_user = random.choice(list(trainset.all_users()))

# # Obtenir les explications pour cet utilisateur
# explanations = model.explain(random_user)

# # Afficher les explications
# for feature, score in sorted(explanations.items(), key=lambda x: x[1], reverse=True):
#     print(f"{feature}: {score:.3f}")

class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        if features_method is None:
            df_features = None
        elif features_method == "title_length": # a naive method that creates only 1 feature based on title length
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')
        else: # (implement other feature creations here)
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features
    

    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)
        
        # Preallocate user profiles
        self.user_profile = {u: None for u in trainset.all_users()}

        if self.regressor_method == 'random_score':
            pass
        
        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]
                
        elif self.regressor_method == 'linear_regression':
            for u in self.user_profile:
                # Créer DataFrame avec les ratings de l'utilisateur
                df_user = pd.DataFrame(
                    self.trainset.ur[u],
                    columns=['inner_item_id', 'rating']
                )
                
                # Convertir les inner_item_id en raw_item_id
                df_user['item_id'] = df_user['inner_item_id'].map(self.trainset.to_raw_iid)
                
                # Fusionner avec les features de contenu
                df_user = df_user.merge(
                    self.content_features,
                    how='left',
                    left_on='item_id',
                    right_index=True
                )
                
                # Extraire les features et targets en arrays numpy
                feature_names = ['n_character_title']
                X = df_user[feature_names].values
                y = df_user['rating'].values
                
                # Créer et entraîner le modèle linéaire sans intercept
                model = LinearRegression(fit_intercept=True)
                model.fit(X, y)
                
                # Sauvegarder le modèle dans le profil utilisateur
                self.user_profile[u] = model
                
        else:
            pass
            # (implement here the regressor fitting)  
        
    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        # First, handle cases for unknown users and items
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')


        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5,5)

        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])

        elif self.regressor_method == 'linear_regression':
            # Convertir l'inner item id en raw item id
            raw_item_id = self.trainset.to_raw_iid(i)
            
            # Récupérer les features de l'item
            item_features = self.content_features.loc[raw_item_id:raw_item_id, :].values
            
            # Faire la prédiction avec le modèle de l'utilisateur
            model = self.user_profile[u]
            score = model.predict(item_features)[0]  # Prendre le premier élément du tableau
            
        
        else:
            score=None
            # (implement here the regressor prediction)

        return score
