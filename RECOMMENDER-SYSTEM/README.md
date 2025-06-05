# 📊 Content-Based Recommender System — MLSMM2156

## Table of Contents
- [📊 Content-Based Recommender System](#-content-based-recommender-system)
- [🎯 Objective](#-objective)
- [📁 Datasets](#-datasets)
- [🔍 Content-Based Approach](#-content-based-approach)
- [📊 Results](#-results)
- [🔍 Evaluation and Known Issues](#-evaluation-and-known-issues)
- [🧪 How to use the notebooks](#-how-to-use-the-notebooks)
- [👨‍💻 To collaborators](#-to-collaborators)
- [📦 Dependencies](#-dependencies)

---

## 🎯 Objective
[(Back to top)](#table-of-contents)

This project implements a content-based recommendation system for movies, developed as part of a Master 1 course in management engineering. The system uses movie features to predict user preferences and generate personalized recommendations.



### Key References
- Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.
- Aggarwal, C. C. (2016). Recommender Systems: The Textbook. Springer.
- Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. IEEE Transactions on Knowledge and Data Engineering.

## 📁 Datasets
[(Back to top)](#table-of-contents)

The project uses two datasets:

1. **MovieLens (Normal Dataset)**
   - 100,000 ratings (1-5)
   - 943 users
   - 1,682 movies
   - Available features: genres, release year, etc.

2. **MovieLens (Hackathon Dataset)**
   - Additional features: genome tags, visual features
   - Same base structure as normal dataset

### Data Structure

The project uses the following data files:

1. **Movies Data**
   - `movies.csv`: Contains movie information
     - `movieId`: Unique identifier for each movie
     - `title`: Movie title with release year
     - `genres`: Pipe-separated list of genres

2. **Ratings Data**
   - `ratings.csv`: Contains user ratings
     - `userId`: Unique identifier for each user
     - `movieId`: Movie identifier
     - `rating`: User rating (1-5)
     - `timestamp`: Rating timestamp

3. **Genome Tags (Hackathon Dataset)**
   - `genome-tags.csv`: Contains tag information
     - `tagId`: Unique identifier for each tag
     - `tag`: Tag name
   - `genome-scores.csv`: Contains tag relevance scores
     - `movieId`: Movie identifier
     - `tagId`: Tag identifier
     - `relevance`: Relevance score (0-1)

4. **Visual Features (Hackathon Dataset)**
   - `visual-features.csv`: Contains visual feature vectors
     - `ML_Id`: Movie identifier
     - `feature_1` to `feature_n`: Visual feature values

5. **Links Data**
   - `links.csv`: Contains external movie IDs
     - `movieId`: Movie identifier
     - `imdbId`: IMDB identifier
     - `tmdbId`: TMDB identifier

6. **Tags Data**
   - `tags.csv`: Contains user-generated tags
     - `userId`: User identifier
     - `movieId`: Movie identifier
     - `tag`: Tag text
     - `timestamp`: Tag timestamp

## 🔍 Content-Based Approach
[(Back to top)](#table-of-contents)

### Features Used

1. **Basic Features**
   - Release year
   - Title length
   - Movie age
   - Decade

2. **Textual Features**
   - TF-IDF on genres
   - Title length

3. **Statistical Features**
   - Genre popularity
   - User statistics (mean, std of ratings)

4. **Additional Features (Hackathon)**
   - Genome tags
   - Visual features

### Regression Models

1. **Linear Regression**
   - Simple and interpretable
   - Fast training

2. **Random Forest**
   - Captures non-linear interactions
   - Robust to outliers

3. **XGBoost**
   - High performance
   - Efficient feature handling

4. **Neural Network**
   - Architecture: 256 -> 128 -> 64 -> 1
   - Dropout for regularization

### Explainability

1. **`.explain(u)` Method**
   - Takes user `u` as input
   - Returns feature importance
   - Scores normalized between 0 and 1

2. **Visualizations**
   - Feature correlation matrix
   - Global feature importance
   - User profiles

## 📊 Results
[(Back to top)](#table-of-contents)

### Comparison of Random Models

| Model         | RMSE    | MAE     | Hit Rate | Novelty    |
|---------------|---------|---------|----------|------------|
| Random Score  | 1.827   | 1.487   | 0.000    | 183,969    |
| Random Sample | 1.370   | 1.019   | 0.003    | 182,859    |

The `Random Sample` model significantly outperforms `Random Score` with an RMSE of 1.370 compared to 1.827. This better performance can be explained by the fact that `Random Sample` draws its predictions from actual user ratings, thus capturing the distribution of user preferences, while `Random Score` generates completely random ratings without considering historical data.

### Linear Regression Model Evaluation

| Configuration     | RMSE    | MAE     | Hit Rate | Novelty    |
|-------------------|---------|---------|----------|------------|
| Without Intercept | 1.561   | 1.295   | 0.006    | 203,343    |
| With Intercept    | 0.976   | 0.761   | 0.001    | 205,708    |

The linear regression model shows significant improvement when using `fit_intercept=True`. The RMSE decreases from 1.561 to 0.976, and the MAE from 1.295 to 0.761. This improvement occurs because the intercept allows the model to account for the baseline rating bias in the dataset, capturing the average rating tendency of users. 

### Normal Dataset Performance

| Features      | Regressor       | RMSE    | MAE    | Time (s) |
|---------------|----------------|---------|--------|----------|
| all_features  | random_forest  | 0.331   | 0.251  | —        |
| all_features  | xgboost        | 0.351   | 0.209  | —        |
| all_features  | neural_network | 0.642   | 0.465  | —        |

### Hackathon Dataset Performance

| Features                 | Regressor         | RMSE    | MAE    | Time (s)   |
|--------------------------|-------------------|---------|--------|------------|
| genome_tags              | linear_regression | 0.719   | 0.546  | 27.43      |
| all_features_with_genome | linear_regression | 0.652   | 0.491  | 57.43      |
| all_features_with_genome | random_forest     | 0.289   | 0.217  | 200.16     |
| all_features_with_genome | xgboost           | 0.184   | 0.112  | 240.76     |
| all_features_with_genome | neural_network    | 0.502   | 0.370  | 483.32     |

The results summarize the performance of different regressors on both the normal and hackathon datasets. 
XGBoost consistently achieved the best results overall. For detailed analysis and visualizations, refer to 
`model_train_and_explain.ipynb` and `visualizations/output.png`

The feature importances are saved in the `visualizations/global_importance_...` files. You can refer to these files for insights into which features contributed most to the model's predictions.


## 🔍 Evaluation and Known Issues
[(Back to top)](#table-of-contents)

### Known Issues

1. **Neural Network Implementation**
   - Potential issues with gradient computation
   - Memory usage spikes during training
   - Long training times for large datasets

2. **Random Forest**
   - High memory consumption
   - Long training times (up to 36 minutes)
   - Potential overfitting on small datasets

3. **XGBoost**
   - GPU memory management issues
   - Potential numerical stability problems

### Performance Considerations

1. **Training Time**
   - Random Forest: ~36 minutes
   - XGBoost: ~20-25 minutes
   - Neural Network: ~30-40 minutes
   - Linear Regression: ~5 minutes

2. **Memory Usage**
   - High memory requirements for feature matrices
   - GPU memory management for neural networks
   - Large model storage for Random Forest

3. **Optimization Opportunities**
   - Batch processing for neural networks
   - Feature selection to reduce dimensionality
   - Model parameter tuning
   - Implement early stopping
   - Use sparse matrices for large feature sets


## 🧪 How to use the notebooks
[(Back to top)](#table-of-contents)

1. **Set up the Data**
   - Download and unzip the dataset
   - Place it in the appropriate directory

2. **Switch dataset size**
   - Open `constants.py` and modify the `DATA_PATH` to switch between datasets
     ```python
     DATA_PATH = Path('datatiny')  # or 'datasmall', 'datatest'
     ```

3. **Run the notebooks in order**
   - `content_based.ipynb`: Main implementation (optional)
   - `feature_analysis.ipynb`: Feature analysis (optional)
   - `evaluator.ipynb`: Model evaluation (optional)
   - `model_train_and_explain.ipynb`: Explainability analysis and performance comparison

4. **Visualization Directory**
   - Create a `visualizations` directory in your workspace
   - All plots and visualizations will be saved there

## 👨‍💻 To collaborators
[(Back to top)](#table-of-contents)

This project is designed for collaborative work in the MLSMM2156 course. This notebook is meant to be executed before developing any recommendation algorithm.  
It ensures that the data is correctly loaded, and helps all members to understand
- The dataset's structure
- Data quality and completeness
- Key metrics (sparsity, rating distribution, etc.)
---

## 📦 Dependencies
[(Back to top)](#table-of-contents)

Required Python packages:
```bash
pip install pandas numpy matplotlib scipy scikit-learn xgboost torch surprise seaborn
```

## Project Structure
```
mlsmm2156/
├── models.py                 # Content-based model implementation
├── models_hackaton.py        # Version with additional features
├── loaders.py               # Data loading utilities
├── constants.py             # Project constants and configurations
├── configs.py               # Model configurations
├── content_based.ipynb      # Content-based model implementation notebook
├── feature_analysis.ipynb   # Feature analysis and visualization
├── evaluator.ipynb          # Model evaluation notebook
├── model_train_and_explain.ipynb  # Explainability visualizations and performance evaluation notebook
├── user_based.ipynb         # User-based collaborative filtering
├── hackathon_make_predictions.ipynb  # Hackathon predictions
```

