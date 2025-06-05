# Requirements pour l'amélioration du système de recommandation

## Objectif
Améliorer le modèle actuel pour obtenir le meilleur RMSE possible dans le cadre du hackathon.

## Étapes d'amélioration du modèle

### 1. Construction de meilleures features

#### 1.1 Intégration de l'année de sortie des films
- [ ] Extraire l'année de sortie de chaque film
- [ ] Créer une feature numérique directe avec l'année
- [ ] Calculer l'âge du film (année actuelle - année de sortie)
- [ ] Créer des variables catégorielles pour les décennies (années 80, 90, 2000, etc.)
- [ ] Normaliser cette feature avec StandardScaler ou MinMaxScaler

#### 1.2 Exploitation des genres de films
- [ ] Encoder les genres avec une représentation one-hot (un vecteur binaire par genre)
- [ ] Appliquer une vectorisation TF-IDF sur les genres pour pondérer leur importance
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  
  # Préparer les données de genre sous forme de chaînes de caractères
  movies['genres_str'] = movies['genres'].apply(lambda x: ' '.join(x.split('|')))
  
  # Appliquer TF-IDF
  tfidf = TfidfVectorizer()
  genres_tfidf = tfidf.fit_transform(movies['genres_str'])
  ```
- [ ] Calculer la note moyenne par genre pour créer des features de popularité de genre
- [ ] Créer des clusters de genres similaires pour réduire la dimensionnalité

#### 1.3 Utilisation du dataset Genome (tags)
- [ ] Charger et explorer le dataset Genome
- [ ] Calculer la pertinence moyenne des tags par film
- [ ] Sélectionner les N tags les plus importants selon leur relevance score
- [ ] Appliquer une réduction de dimensionnalité (PCA ou t-SNE) sur les vecteurs de tags
  ```python
  from sklearn.decomposition import PCA
  
  # Réduire la dimensionnalité des tags
  pca = PCA(n_components=20)
  genome_features_reduced = pca.fit_transform(genome_features)
  ```
- [ ] Créer des features de similarité entre les préférences utilisateur et les tags

#### 1.4 Intégration du dataset Visuals
- [ ] Extraire les caractéristiques visuelles principales (couleurs, formes, etc.)
- [ ] Gérer les valeurs manquantes:
  ```python
  # Option 1: Imputation par KNN
  from sklearn.impute import KNNImputer
  imputer = KNNImputer(n_neighbors=5)
  visuals_imputed = imputer.fit_transform(visuals_features)
  
  # Option 2: Création d'un indicateur de données manquantes
  movies['has_visual_data'] = movies['visual_id'].notna().astype(int)
  ```
- [ ] Appliquer une normalisation sur les features visuelles
- [ ] Créer des clusters de films visuellement similaires

### 2. Choix d'un meilleur régresseur

#### 2.1 Test de différents modèles
- [ ] Implémenter un Random Forest Regressor
  ```python
  from sklearn.ensemble import RandomForestRegressor
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  ```
- [ ] Tester un Gradient Boosting (XGBoost ou LightGBM)
  ```python
  import xgboost as xgb
  xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
  xgb_model.fit(X_train, y_train)
  ```
- [ ] Essayer un Support Vector Regressor avec différents kernels
  ```python
  from sklearn.svm import SVR
  svr_model = SVR(kernel='rbf')
  svr_model.fit(X_train, y_train)
  ```
- [ ] Tester des modèles de factorisation matricielle spécifiques aux systèmes de recommandation
  ```python
  from surprise import SVD
  from surprise import Dataset, Reader
  
  # Préparer les données au format surprise
  reader = Reader(rating_scale=(0.5, 5))
  data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
  
  # Utiliser SVD pour la factorisation matricielle
  svd_model = SVD(n_factors=100, random_state=42)
  ```

#### 2.2 Optimisation des hyperparamètres
- [ ] Mettre en place une recherche par grid search pour les hyperparamètres
  ```python
  from sklearn.model_selection import GridSearchCV
  
  # Exemple pour Random Forest
  param_grid = {
      'n_estimators': [50, 100, 200],
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10]
  }
  
  grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                            param_grid, 
                            cv=5, 
                            scoring='neg_mean_squared_error')
  grid_search.fit(X_train, y_train)
  best_rf_model = grid_search.best_estimator_
  ```
- [ ] Utiliser une validation croisée pour éviter le surajustement
- [ ] Comparer les performances avec l'évaluateur fourni

### 3. Sélection et pondération des features

#### 3.1 Sélection des features les plus pertinentes
- [ ] Appliquer Recursive Feature Elimination (RFE)
  ```python
  from sklearn.feature_selection import RFE
  
  # Sélectionner les meilleures features
  selector = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=20)
  X_selected = selector.fit_transform(X_train, y_train)
  ```
- [ ] Utiliser l'importance des features d'un Random Forest
  ```python
  rf = RandomForestRegressor(random_state=42)
  rf.fit(X_train, y_train)
  
  # Récupérer et visualiser l'importance des features
  feature_importances = pd.DataFrame({
      'feature': feature_names,
      'importance': rf.feature_importances_
  }).sort_values('importance', ascending=False)
  ```
- [ ] Tester une méthode basée sur la corrélation (comme SelectKBest avec r2_score)

#### 3.2 Techniques avancées
- [ ] Implémenter un stacking de modèles
  ```python
  from sklearn.ensemble import StackingRegressor
  
  estimators = [
      ('rf', RandomForestRegressor(random_state=42)),
      ('xgb', xgb.XGBRegressor(random_state=42)),
      ('svr', SVR())
  ]
  
  stacking_regressor = StackingRegressor(
      estimators=estimators,
      final_estimator=LinearRegression()
  )
  
  stacking_regressor.fit(X_train, y_train)
  ```
- [ ] Créer un ensemble de modèles par vote ou moyenne
- [ ] Combiner différentes approches (content-based et collaborative filtering)

### 4. Techniques spécifiques aux systèmes de recommandation

#### 4.1 Features basées sur les utilisateurs
- [ ] Calculer des statistiques par utilisateur (moyenne, écart-type des notes)
- [ ] Identifier les biais d'utilisateur (certains notent systématiquement plus haut/bas)
- [ ] Créer des clusters d'utilisateurs et utiliser l'appartenance au cluster comme feature

#### 4.2 Features d'interaction
- [ ] Créer des features d'interaction entre l'utilisateur et le film
- [ ] Calculer la similarité entre le profil d'utilisateur et les caractéristiques du film
- [ ] Incorporer l'historique des notes de l'utilisateur comme contexte

#### 4.3 Features temporelles
- [ ] Analyser les tendances temporelles dans les notes
- [ ] Créer des features liées au moment où la note a été donnée
- [ ] Détecter les changements de préférence au fil du temps

## Workflow d'expérimentation

1. Commencer par implémenter les features de base (année et genres)
2. Tester rapidement différents modèles avec ces features
3. Ajouter progressivement des features plus complexes
4. Optimiser les hyperparamètres du modèle le plus prometteur
5. Soumettre régulièrement des prédictions pour suivre l'amélioration du RMSE
6. Documenter systématiquement les résultats de chaque expérience

## Évaluation et validation

- [ ] Utiliser une validation croisée k-fold pour évaluer la robustesse du modèle
- [ ] Comparer les modèles avec l'évaluateur fourni
- [ ] Analyser les types d'erreurs (sur quels films/utilisateurs le modèle se trompe-t-il?)
- [ ] Vérifier qu'il n'y a pas de fuite de données entre l'entraînement et la validation

## Bonnes pratiques

- Normaliser toutes les features numériques avant l'entraînement
- Éviter le surajustement en utilisant la validation croisée
- Surveiller les temps de calcul des modèles plus complexes
- Sauvegarder régulièrement les meilleurs modèles
- Documenter les expériences et les résultats pour faciliter l'itération
