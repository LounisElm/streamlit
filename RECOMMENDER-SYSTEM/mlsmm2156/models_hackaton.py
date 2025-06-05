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
        
        
class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method, nn_config=None):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.features_method = features_method
        self.nn_config = nn_config
        self.user_profile_explain = {}  # Dictionnaire pour stocker les explications
        
        # Initialiser le tfidf_vectorizer AVANT d'appeler create_content_features
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=genre_tokenizer)

        
        # Initialiser le device pour le réseau de neurones
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Configuration de l'entraînement
        self.training_config = {
            'batch_size': 32,
            'epochs': 50,
            'early_stopping_patience': 5,
            'learning_rate': 0.001
        }
        
        # Initialiser les features
        self.content_features = self.create_content_features(features_method)
        
        if regressor_method == "neural_network":
            self.model = None  # Sera initialisé dans fit()
            self.optimizer = None
            self.criterion = nn.MSELoss()
    
    def create_content_features(self, features_method):
        def align_on_movie_ids(df, reference_index):
            df = df.loc[df.index.isin(reference_index)]
            return df.reindex(reference_index)

        df_items = load_items()
        df_ratings = load_ratings()

        if features_method == "genome_tags":
            genome_scores, genome_tags_df = load_genome_data()
            if genome_scores is None or genome_tags_df is None:
                raise ValueError("Genome dataset not available")

            movie_tag_relevance = genome_scores.groupby('movieId')['relevance'].mean().reset_index()
            movie_tag_relevance.columns = ['movieId', 'avg_tag_relevance']

            top_tags_ids = genome_scores.groupby('tagId')['relevance'].mean().nlargest(100).index

            tag_matrix = pd.pivot_table(
                genome_scores[genome_scores['tagId'].isin(top_tags_ids)],
                values='relevance',
                index='movieId',
                columns='tagId',
                fill_value=0
            )

            n_samples, n_features = tag_matrix.shape
            n_pca_components = min(20, n_samples, n_features)

            if n_pca_components < 1:
                print(f"Warning: Not enough data for PCA on genome_tags. Skipping PCA and KMeans.")
                tag_features_df = pd.DataFrame(index=tag_matrix.index)
                tag_clusters_df = pd.DataFrame(index=tag_matrix.index)
            else:
                pca = PCA(n_components=n_pca_components)
                tag_features_reduced = pca.fit_transform(tag_matrix)
                tag_features_df = pd.DataFrame(
                    tag_features_reduced,
                    index=tag_matrix.index,
                    columns=[f'tag_pca_{i}' for i in range(n_pca_components)]
                )

                inertias = []
                K_range = range(2, 21)
                for k in K_range:
                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    kmeans_temp.fit(tag_features_reduced)
                    inertias.append(kmeans_temp.inertia_)

                diffs = np.diff(inertias)
                diffs_r = np.diff(diffs)
                optimal_k_tags = K_range[np.argmax(diffs_r) + 1]

                kmeans = KMeans(n_clusters=optimal_k_tags, random_state=42, n_init='auto')
                tag_clusters = kmeans.fit_predict(tag_features_reduced)
                tag_clusters_df = pd.DataFrame(
                    tag_clusters,
                    index=tag_matrix.index,
                    columns=['tag_cluster']
                )

            df_features = pd.concat([
                movie_tag_relevance.set_index('movieId'),
                tag_features_df,
                tag_clusters_df
            ], axis=1).reindex(df_items.index)

            return df_features

        elif features_method == "genome_user_preferences":
            genome_scores, genome_tags = load_genome_data()
            if genome_scores is None or genome_tags is None:
                raise ValueError("Genome dataset not available")

            user_tag_preferences = []
            for user_id in df_ratings['userId'].unique():
                user_movies = df_ratings[df_ratings['userId'] == user_id]['movieId']
                user_movie_tags = genome_scores[genome_scores['movieId'].isin(user_movies)]
                tag_means = user_movie_tags.groupby('tagId')['relevance'].mean()
                for tag_id, relevance in tag_means.items():
                    user_tag_preferences.append({
                        'userId': user_id,
                        'tagId': tag_id,
                        'preference': relevance
                    })

            user_tag_preferences_df = pd.DataFrame(user_tag_preferences)
            user_tag_matrix = pd.pivot_table(
                user_tag_preferences_df,
                values='preference',
                index='userId',
                columns='tagId',
                fill_value=0
            )

            pca = PCA(n_components=20)
            user_tag_features_reduced = pca.fit_transform(user_tag_matrix)
            user_tag_features_df = pd.DataFrame(
                user_tag_features_reduced,
                index=user_tag_matrix.index,
                columns=[f'user_tag_pca_{i}' for i in range(20)]
            )

            kmeans = KMeans(n_clusters=15, random_state=42)
            user_tag_clusters = kmeans.fit_predict(user_tag_features_reduced)
            user_tag_clusters_df = pd.DataFrame(
                user_tag_clusters,
                index=user_tag_matrix.index,
                columns=['user_tag_cluster']
            )

            df_features = pd.concat([user_tag_features_df, user_tag_clusters_df], axis=1)
            return df_features

        elif features_method == "all_features_with_genome":
            base_features = self.create_content_features("all_features")
            genome_features = self.create_content_features("genome_tags")

            df_features = pd.concat([
                base_features,
                genome_features,
            ], axis=1)

            df_features = df_features.loc[:, ~df_features.columns.duplicated()]
            return df_features

        elif features_method == "all_features_with_genome_and_visuals":
            base_features = self.create_content_features("all_features_with_genome")
            visual_features = self.create_content_features("visual_features")

            df_features = pd.concat([
                base_features,
                visual_features
            ], axis=1)

            df_features = df_features.loc[:, ~df_features.columns.duplicated()]
            return df_features

        elif features_method == "visual_features":
            hack = C.HACK_PATH / 'content' if C.HACK_PATH.exists() else C.CONTENT_PATH
            C.CONTENT_PATH = hack
            visuals_path = os.path.join(C.CONTENT_PATH, "visuals\\VisualFeatures13K_Log.csv")
            visuals_data = pd.read_csv(visuals_path)
            visuals_data.rename(columns={'ML_ID': 'movieId'}, inplace=True)

            imputer = KNNImputer(n_neighbors=5)
            visuals_imputed = imputer.fit_transform(visuals_data.select_dtypes(include=[np.number]))

            scaler = StandardScaler()
            visuals_scaled = scaler.fit_transform(visuals_imputed)

            visuals_df = pd.DataFrame(
                visuals_scaled,
                index=visuals_data['movieId'],
                columns=[f'visual_feature_{i}' for i in range(visuals_scaled.shape[1])]
            )

            return visuals_df

        elif features_method == "all_features":
            features_list = []

            features_list.append(df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title'))

            year_df = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)', expand=False).to_frame('title_year')
            year_df['title_year'] = pd.to_numeric(year_df['title_year'], errors='coerce')
            current_year = pd.Timestamp.now().year
            year_df['movie_age'] = current_year - year_df['title_year']
            features_list.append(year_df)

            decade_df = pd.get_dummies((year_df['title_year'] // 10) * 10, prefix='decade')
            features_list.append(decade_df)

            genre_ratings = []
            for _, row in df_items.iterrows():
                if pd.notna(row[C.GENRES_COL]) and row[C.GENRES_COL] != '(no genres listed)':
                    genres = row[C.GENRES_COL].split('|')
                    movie_ratings = df_ratings[df_ratings['movieId'] == row.name]['rating']
                    if not movie_ratings.empty:
                        for genre in genres:
                            genre_ratings.append({'genre': genre, 'rating': movie_ratings.mean()})

            genre_ratings_df = pd.DataFrame(genre_ratings)
            genre_means = genre_ratings_df.groupby('genre')['rating'].mean()

            genre_pop_df = pd.DataFrame(index=df_items.index)
            for genre in genre_means.index:
                genre_pop_df[f'genre_{genre}_popularity'] = df_items[C.GENRES_COL].apply(
                    lambda x: genre_means[genre] if pd.notna(x) and genre in x.split('|') else 0
                )
            features_list.append(genre_pop_df)

            user_stats = df_ratings.groupby('userId').agg({
                'rating': ['mean', 'std', 'count']
            }).reset_index()
            user_stats.columns = ['userId', 'user_mean_rating', 'user_rating_std', 'user_rating_count']
            global_mean = df_ratings['rating'].mean()
            user_stats['user_bias'] = user_stats['user_mean_rating'] - global_mean

            movie_user_stats = []
            for movie_id in df_items.index:
                movie_ratings = df_ratings[df_ratings['movieId'] == movie_id]
                if not movie_ratings.empty:
                    merged = movie_ratings.merge(user_stats, on='userId')
                    movie_user_stats.append({
                        'movieId': movie_id,
                        'avg_user_mean': merged['user_mean_rating'].mean(),
                        'avg_user_std': merged['user_rating_std'].mean(),
                        'avg_user_bias': merged['user_bias'].mean()
                    })

            user_stats_df = pd.DataFrame(movie_user_stats).set_index('movieId')
            features_list.append(user_stats_df)

            genres_processed = df_items[C.GENRES_COL].fillna('').replace('(no genres listed)', '')
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(genres_processed)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_items.index, columns=[f"tfidf_{name}" for name in feature_names])
            features_list.append(tfidf_df)

            df_features = pd.concat(features_list, axis=1)
            return df_features

        else:
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')    
    

    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)
        
        # Preallocate user profiles
        self.user_profile = {u: None for u in trainset.all_users()}
        self.user_profile_explain = {u: {} for u in trainset.all_users()}

        if self.content_features is None or self.content_features.empty:
            return

        feature_names = self.content_features.columns.tolist()
        if not feature_names:
            return

        for u_inner_id in self.trainset.all_users():
            user_ratings_data = self.trainset.ur[u_inner_id]
            if not user_ratings_data:
                continue

            # Préparer les données pour l'utilisateur
            df_user_ratings = pd.DataFrame(user_ratings_data, columns=['inner_item_id', 'user_ratings'])
            df_user_ratings['item_id'] = df_user_ratings['inner_item_id'].apply(lambda x: self.trainset.to_raw_iid(x))

            df_merged = df_user_ratings.merge(
                self.content_features,
                how='left',
                left_on='item_id',
                right_index=True
            )
            
            df_merged_cleaned = df_merged.dropna(subset=feature_names + ['user_ratings'])

            if df_merged_cleaned.empty or len(df_merged_cleaned) < 2:
                continue

            X_user = df_merged_cleaned[feature_names].values
            y_user = df_merged_cleaned['user_ratings'].values

            # Calculer les explications basées sur les notes pondérées
            total_rating = sum(y_user)
            if total_rating > 0:
                for feature in feature_names:
                    weighted_sum = 0
                    for idx, rating in enumerate(y_user):
                        weighted_sum += rating * X_user[idx, feature_names.index(feature)]
                    self.user_profile_explain[u_inner_id][feature] = weighted_sum / total_rating

            # Entraîner le modèle selon la méthode choisie
            if self.regressor_method == 'random_score':
                self.user_profile[u_inner_id] = None

            elif self.regressor_method == 'random_sample':
                self.user_profile[u_inner_id] = ([rating for _, rating in user_ratings_data], None)

            elif self.regressor_method in ['linear_regression', 'random_forest', 'xgboost']:
                scaler = StandardScaler()
                try:
                    X_user_scaled = scaler.fit_transform(X_user)
                except ValueError:
                    self.user_profile[u_inner_id] = None
                    continue

                try:
                    if self.regressor_method == 'linear_regression':
                        model = LinearRegression(fit_intercept=True)
                        model.fit(X_user_scaled, y_user)
                        # Pour la régression linéaire, utiliser les coefficients comme explications
                        for feature, coef in zip(feature_names, model.coef_):
                            self.user_profile_explain[u_inner_id][feature] = abs(coef)
                    elif self.regressor_method == 'random_forest':
                        model = RandomForestRegressor(
                            n_estimators=100, 
                            random_state=0, 
                            n_jobs=-1, 
                            max_features='sqrt'
                        )
                        model.fit(X_user_scaled, y_user)
                        # Pour RandomForest, utiliser les importances des features
                        for feature, importance in zip(feature_names, model.feature_importances_):
                            self.user_profile_explain[u_inner_id][feature] = importance
                    elif self.regressor_method == 'xgboost':
                        model = xgb.XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=0,
                            n_jobs=-1
                        )
                        model.fit(X_user_scaled, y_user)
                        # Pour XGBoost, utiliser les importances des features
                        for feature, importance in zip(feature_names, model.feature_importances_):
                            self.user_profile_explain[u_inner_id][feature] = importance

                    self.user_profile[u_inner_id] = (model, scaler)
                except Exception as e:
                    print(f"Error fitting model for user {u_inner_id}: {str(e)}")
                    self.user_profile[u_inner_id] = None

            elif self.regressor_method == "neural_network":
                # Normaliser les features
                scaler = StandardScaler()
                X_user_scaled = scaler.fit_transform(X_user)
                
                # Convertir en tenseurs PyTorch
                X_tensor = torch.FloatTensor(X_user_scaled).to(self.device)
                y_tensor = torch.FloatTensor(y_user).reshape(-1, 1).to(self.device)
                
                # Initialiser le modèle
                model = MovieRecommenderNN(len(feature_names)).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Entraînement
                model.train()
                for epoch in range(100):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = self.criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Pour le réseau de neurones, utiliser les poids de la première couche comme explications
                first_layer_weights = model.network[0].weight.data.abs().mean(dim=0)
                for feature, weight in zip(feature_names, first_layer_weights):
                    self.user_profile_explain[u_inner_id][feature] = weight.item()
                
                self.user_profile[u_inner_id] = (model, scaler)
            else:
                raise NotImplementedError(f"Regressor method {self.regressor_method} not implemented in fit.")
    
    def explain(self, u):
        """
        Retourne l'importance de chaque feature pour l'utilisateur u.
        Les scores sont normalisés entre 0 et 1.
        
        Args:
            u: ID de l'utilisateur
            
        Returns:
            dict: Dictionnaire {feature_name: feature_score} où feature_score ∈ [0, 1]
        """
        if not self.trainset.knows_user(u):
            raise PredictionImpossible('User is unknown.')
        
        if u not in self.user_profile_explain:
            return {}
            
        # Récupérer les explications brutes
        raw_explanations = self.user_profile_explain[u]
        
        # Normaliser les scores entre 0 et 1
        if raw_explanations:
            max_score = max(raw_explanations.values())
            min_score = min(raw_explanations.values())
            if max_score != min_score:  # Éviter la division par zéro
                normalized_explanations = {
                    feature: (score - min_score) / (max_score - min_score)
                    for feature, score in raw_explanations.items()
                }
            else:
                normalized_explanations = {feature: 0.5 for feature in raw_explanations}
        else:
            normalized_explanations = {}
            
        return normalized_explanations
    

    def estimate(self, u, i): # u is inner user ID, i is inner item ID
        """Scoring component used for item filtering"""
        if not self.trainset.knows_user(u): 
             raise PredictionImpossible(f'User with inner ID {u} is unkown.')
        
        if not self.trainset.knows_item(i):
            raise PredictionImpossible(f'Item with inner ID {i} is unkown.')

        profile_data = self.user_profile.get(u)

        if self.regressor_method == 'random_score':
            score = rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])

        elif self.regressor_method == 'random_sample':
            if profile_data and profile_data[0] and len(profile_data[0]) > 0:
                score = rd.choice(profile_data[0])
            else: 
                 raise PredictionImpossible(f'No ratings available to sample for user {self.trainset.to_raw_uid(u)}')
        
        elif self.regressor_method in ['linear_regression', 'random_forest', 'xgboost']:
            if profile_data is None:
                raise PredictionImpossible(f'No model/scaler for user {self.trainset.to_raw_uid(u)}.')
            
            model, scaler = profile_data
            if model is None or scaler is None: # Should be caught by profile_data is None, but as safeguard
                 raise PredictionImpossible(f'Model or scaler is None for user {self.trainset.to_raw_uid(u)}.')

            raw_item_id = self.trainset.to_raw_iid(i) # Convert inner item ID 'i' to raw item ID

            if self.content_features is None or raw_item_id not in self.content_features.index:
                 raise PredictionImpossible(f'Item {raw_item_id} not in content_features or content_features is None.')

            feature_names = self.content_features.columns.tolist()
            if not feature_names: # Should not happen if fit was successful and content_features were generated
                raise PredictionImpossible('No feature names found in content_features during estimation.')

            item_features_series = self.content_features.loc[raw_item_id, feature_names]
            
            # Check for NaNs in the specific item's features
            if item_features_series.isna().any():
                raise PredictionImpossible(f'One or more features are NaN for item {raw_item_id}. Features: {feature_names}, Values: {item_features_series.values}')

            X_item = item_features_series.values.reshape(1, -1) # Reshape for single prediction
            
            try:
                X_item_scaled = scaler.transform(X_item)
                score = model.predict(X_item_scaled)[0]
                min_rating, max_rating = self.trainset.rating_scale
                score = np.clip(score, min_rating, max_rating)
            except Exception as e:
                # If scaler was fit on features that are now gone, or other issues.
                # Also, if X_item contains values outside the range scaler was fit on (e.g. a new genre for TFIDF if not handled)
                raise PredictionImpossible(f'Error predicting with {self.regressor_method} for user {self.trainset.to_raw_uid(u)}, item {raw_item_id}: {e}')
        elif self.regressor_method == "neural_network":
            if profile_data is None:
                raise PredictionImpossible(f'No model/scaler for user {self.trainset.to_raw_uid(u)}.')
            
            model, scaler = profile_data
            if model is None or scaler is None:
                raise PredictionImpossible(f'Model or scaler is None for user {self.trainset.to_raw_uid(u)}.')

            raw_item_id = self.trainset.to_raw_iid(i)
            
            if self.content_features is None or raw_item_id not in self.content_features.index:
                raise PredictionImpossible(f'Item {raw_item_id} not in content_features or content_features is None.')

            feature_names = self.content_features.columns.tolist()
            item_features_series = self.content_features.loc[raw_item_id, feature_names]
            
            if item_features_series.isna().any():
                raise PredictionImpossible(f'One or more features are NaN for item {raw_item_id}.')

            X_item = item_features_series.values.reshape(1, -1)
            X_item_scaled = scaler.transform(X_item)
            
            # Convertir en tenseur et faire la prédiction
            X_tensor = torch.FloatTensor(X_item_scaled).to(self.device)
            model.eval()
            with torch.no_grad():
                score = model(X_tensor).item()
            
            # Clipper le score dans la plage de notes valide
            min_rating, max_rating = self.trainset.rating_scale
            score = np.clip(score, min_rating, max_rating)
            
            return score
        else:
            raise NotImplementedError(f"Regressor method {self.regressor_method} not implemented in estimate.")
            
        return score

    def optimize_hyperparameters(self, X_train, y_train, regressor_method):
        """Optimise les hyperparamètres du régresseur choisi"""
        if regressor_method == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestRegressor(random_state=42)
        
        elif regressor_method == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        else:
            return None  # Pas d'optimisation pour les autres méthodes
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def select_features(self, X, y, n_features=20):
        """Sélectionne les features les plus importantes"""
        # Utiliser RandomForest comme estimateur de base
        base_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Créer le sélecteur RFE
        selector = RFE(
            estimator=base_estimator,
            n_features_to_select=n_features,
            step=1
        )
        
        # Ajuster le sélecteur
        selector.fit(X, y)
        
        # Retourner les features sélectionnées
        return selector.support_

    
    def visualize_user_profile(self, user_id, top_n=10, save_path=None):
        """
        Visualise le profil d'un utilisateur en montrant les features les plus importantes.
        
        Args:
            user_id: ID de l'utilisateur
            top_n: Nombre de features à afficher
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        if not self.trainset.knows_user(user_id):
            raise PredictionImpossible('User is unknown.')
        
        explanations = self.explain(user_id)
        if not explanations:
            print(f"Aucune explication disponible pour l'utilisateur {user_id}")
            return
        
        # Trier les features par importance
        sorted_features = sorted(explanations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_features)
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance de la feature')
        plt.title(f'Profil utilisateur {user_id} - Top {top_n} features')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def visualize_feature_importance_comparison(self, user_ids, top_n=10, save_path=None):
        """
        Compare l'importance des features entre plusieurs utilisateurs.
        
        Args:
            user_ids: Liste d'IDs d'utilisateurs
            top_n: Nombre de features à afficher
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        # Vérifier que tous les utilisateurs existent
        for user_id in user_ids:
            if not self.trainset.knows_user(user_id):
                raise PredictionImpossible(f'User {user_id} is unknown.')
        
        # Obtenir les explications pour chaque utilisateur
        user_explanations = {}
        for user_id in user_ids:
            explanations = self.explain(user_id)
            if explanations:
                # Trier et garder les top_n features
                sorted_features = sorted(explanations.items(), key=lambda x: x[1], reverse=True)[:top_n]
                user_explanations[user_id] = dict(sorted_features)
        
        if not user_explanations:
            print("Aucune explication disponible pour les utilisateurs sélectionnés")
            return
        
        # Créer le graphique
        plt.figure(figsize=(15, 8))
        
        # Préparer les données pour le graphique
        features = list(next(iter(user_explanations.values())).keys())
        x = np.arange(len(features))
        width = 0.8 / len(user_ids)
        
        # Tracer les barres pour chaque utilisateur
        for i, (user_id, explanations) in enumerate(user_explanations.items()):
            scores = [explanations.get(f, 0) for f in features]
            plt.bar(x + i*width, scores, width, label=f'User {user_id}')
        
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Comparaison des profils utilisateurs')
        plt.xticks(x + width*(len(user_ids)-1)/2, features, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def visualize_feature_correlation(self, user_id, save_path=None):
        """
        Visualise la corrélation entre les features pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        if not self.trainset.knows_user(user_id):
            raise PredictionImpossible('User is unknown.')
        
        # Obtenir les données de l'utilisateur
        user_ratings = self.trainset.ur[user_id]
        if not user_ratings:
            print(f"Aucune donnée disponible pour l'utilisateur {user_id}")
            return
        
        # Créer un DataFrame avec les features et les notes
        df_user = pd.DataFrame(user_ratings, columns=['inner_item_id', 'rating'])
        df_user['item_id'] = df_user['inner_item_id'].apply(lambda x: self.trainset.to_raw_iid(x))
        
        # Fusionner avec les features
        df_merged = df_user.merge(
            self.content_features,
            how='left',
            left_on='item_id',
            right_index=True
        )
        
        # Calculer la matrice de corrélation
        corr_matrix = df_merged.corr()
        
        # Créer le graphique
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Matrice de corrélation des features pour l\'utilisateur {user_id}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_global_feature_importance(self, top_n=10):
        """
        Calcule et retourne les features les plus importantes globalement
        
        Args:
            top_n (int): Nombre de features à retourner
            
        Returns:
            dict: Dictionnaire des features les plus importantes avec leurs scores moyens
        """
        if not self.user_profile_explain:
            return {}
            
        # Agréger les importances des features pour tous les utilisateurs
        feature_importances = defaultdict(list)
        for user_explanations in self.user_profile_explain.values():
            for feature, importance in user_explanations.items():
                feature_importances[feature].append(importance)
        
        # Calculer la moyenne des importances pour chaque feature
        global_importances = {
            feature: np.mean(importances) 
            for feature, importances in feature_importances.items()
        }
        
        # Trier les features par importance
        sorted_features = sorted(
            global_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return dict(sorted_features[:top_n])

    def visualize_global_feature_importance(self, top_n=10, save_path=None):
        """
        Visualise l'importance globale des features.
        
        Args:
            top_n: Nombre de features à afficher
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        global_importances = self.get_global_feature_importance(top_n)
        if not global_importances:
            print("Aucune donnée d'importance globale disponible")
            return
        
        features, scores = zip(*global_importances.items())
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance moyenne')
        plt.title(f'Importance globale des features - Top {top_n}')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

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

# # Test avec différentes méthodes de features
# methods = [
#     ("genome_tags", "linear_regression"),
#     ("all_features_with_genome", "random_forest"),
#     ("all_features_with_genome","xgboost"),
#     ("all_features_with_genome", "neural_network"),
#     ("all_features_with_genome_and_visuals","random_forest"),
#     ("all_features_with_genome_and_visuals","xgboost"),
#     ("all_features_with_genome_and_visuals","neural_network")
# ]

# for features_method, regressor_method in methods:
#     print(f"\n{'='*50}")
#     print(f"Test avec {features_method} et {regressor_method}")
#     print(f"{'='*50}")
    
#     # Créer et entraîner le modèle
#     model = ContentBased(features_method=features_method, regressor_method=regressor_method)
#     model.fit(trainset)
    
#     # Afficher les features globales les plus importantes
#     print("\nFeatures globales les plus importantes :")
#     global_importances = model.get_global_feature_importance(top_n=10)
#     for feature, importance in global_importances.items():
#         print(f"{feature}: {importance:.4f}")
    
#     # Afficher les explications pour un utilisateur spécifique
#     user_id = 14
#     print(f"\nExplications pour l'utilisateur {user_id}:")
#     explanations = model.explain(user_id)
#     if explanations:
#         sorted_explanations = sorted(explanations.items(), key=lambda x: x[1], reverse=True)
#         for feature, score in sorted_explanations[:10]:
#             print(f"{feature}: {score:.4f}")
#     else:
#         print("Aucune explication disponible pour cet utilisateur")