# 🎯 Plan d'Action pour le Système de Recommandation Content-Based

## 📋 État des Lieux
- Système de recommandation content-based déjà implémenté avec plusieurs fonctionnalités
- Deux datasets à traiter : MovieLens (normal) et Hackathon
- Focus sur l'explicabilité des prédictions (point 9)
- Niveau Master 1 en ingénierie de gestion

## 🗂️ Structure du Projet à Finaliser

### 1. Implémentation de l'Explicabilité (Point 9)
- [x] Méthode `user_profile_explain` déjà implémentée
- [x] Calcul de la moyenne pondérée des features normalisées par utilisateur
- [x] Méthode `.explain(u)` retournant `{feature_name: score}`
- [ ] Amélioration de l'explicabilité
  - [ ] Ajouter des visualisations des profils utilisateurs
  - [ ] Créer des graphiques d'importance des features
  - [ ] Documenter les patterns d'explicabilité observés

### 2. Optimisation du Modèle
- [x] Features déjà implémentées :
  - [x] Année de sortie
  - [x] TF-IDF pour les genres
  - [x] Données genome
  - [x] Données visuelles (avec gestion des NaN)
- [x] Régresseurs déjà implémentés :
  - [x] Linear Regression
  - [x] Random Forest
  - [x] XGBoost
  - [x] Neural Network
- [ ] Améliorations possibles, peut =être pas nécessaire
  - [ ] Optimisation des hyperparamètres existants
  - [ ] Test de nouvelles combinaisons de features
  - [ ] Analyse de la performance par type de contenu

### 3. Évaluation et Comparaison
- [ ] Évaluation sur le dataset MovieLens et Hackaton
  - [ ] Calcul du RMSE pour chaque combinaison features/régresseur


### 4. Documentation et README (à faire en dernier) Attention lire le @README.MD avant pour adapter
- [ ] Adapter le README.md pour refléter le niveau Master 1
  - [ ] Introduction académique avec références théoriques
  - [ ] Description détaillée des datasets
  - [ ] Explication de l'approche content-based
  - [ ] Documentation des features utilisées
  - [ ] Présentation des modèles de régression
  - [ ] Section sur l'explicabilité des recommandations
  - [ ] Résultats comparatifs sur les deux datasets
  - [ ] Instructions d'exécution

### 5. Finalisation
- [ ] Nettoyage et organisation du code
  - [ ] Vérifier la cohérence entre `models.py` et `models_hackaton.py`
  - [ ] Optimiser les imports
  - [ ] Ajouter des commentaires explicatifs pas nécessaires ou très court et en anglais 
- [ ] Tests unitaires et d'intégration
- [ ] Préparation de la présentation des résultats

## 📊 Livrables Attendus
1. Code source optimisé et documenté
2. Résultats d'évaluation sur les deux datasets
3. Documentation de l'explicabilité des recommandations
4. README.md adapté au niveau Master 1
5. Présentation des résultats et conclusions



