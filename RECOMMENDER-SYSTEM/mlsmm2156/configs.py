# local imports
from models import *


class EvalConfig:
    
    models = [
        # ("baseline_1", ModelBaseline1, {}),
        # ("baseline 2", ModelBaseline2, {}),
        # ("baseline 3", ModelBaseline3, {}),
        # ("baseline 4", ModelBaseline4, {}),
        # ("Rand_score", ContentBased, {"features_method": "title_length", "regressor_method": "random_score"}),
        #("Rand_sample", ContentBased, {"features_method": "title_length", "regressor_method": "random_sample"}),
        ("LinReg", ContentBased, {"features_method": "title_length", "regressor_method": "linear_regression"}),
        # ("CB_NumGenres_RandScore", ContentBased, {"features_method": "num_genres", "regressor_method": "random_score"}), # Renamed for clarity
        # ("CB_AllNum_LinReg", ContentBased, {"features_method": "all_numerical", "regressor_method": "linear_regression"}), # Renamed for clarity
        # # New configurations for TF-IDF and RandomForest
        # ("CB_TFIDF_LinReg", ContentBased, {"features_method": "genres_tfidf", "regressor_method": "linear_regression"}),
        # ("CB_AllNum_RF", ContentBased, {"features_method": "all_numerical", "regressor_method": "random_forest"}),
        # ("CB_TFIDF_RF", ContentBased, {"features_method": "genres_tfidf", "regressor_method": "random_forest"}),
        # ("CB_MovieAge_RF", ContentBased, {"features_method": "movie_age", "regressor_method": "random_forest"}),
        # ("CB_Decade_RF", ContentBased, {"features_method": "decade", "regressor_method": "random_forest"}),
        # ("CB_GenrePopularity_RF", ContentBased, {"features_method": "genre_popularity", "regressor_method": "random_forest"}),
        # ("CB_UserStats_RF", ContentBased, {"features_method": "user_stats", "regressor_method": "random_forest"}),
        # ("CB_AllFeatures_RF", ContentBased, {"features_method": "all_features", "regressor_method": "random_forest"}),
        # # Modèles avec features Genome
        # ("CB_GenomeTags_RF", ContentBased, {
        #     "features_method": "genome_tags",
        #     "regressor_method": "random_forest"
        # }),
        # # ("CB_GenomeUserPrefs_RF", ContentBased, {
        # #     "features_method": "genome_user_preferences",
        # #     "regressor_method": "random_forest"
        # # }),
        # ("CB_AllFeaturesWithGenome_RF", ContentBased, {
        #     "features_method": "all_features_with_genome",
        #     "regressor_method": "random_forest"
        # }),
    ]

    split_metrics = ["MAE","RMSE"]
    loo_metrics = ["hit_rate"]
    full_metrics = ["novelty"]

    # Split parameters
    test_size = 0.2  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
