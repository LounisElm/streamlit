# third parties imports
from pathlib import Path
from surprise import Dataset, SVD, Reader
import platform

class Constant:

    # Check the operating system
    if platform.system() == 'Darwin':
        DATA_PATH = Path(r"C:\Users\Luca\Desktop\Majeur-BA\RECOMMENDER-SYSTEM\mlsmm2156\data\tiny")
        # DATA_PATH = Path("/Users/nicol/Documents/GitHub/Majeur-BA/RECOMMENDER-SYSTEM/mlsmm2156/data/hackathon")
    HACK_PATH = Path(r"C:\Users\Luca\Desktop\Majeur-BA\RECOMMENDER-SYSTEM\mlsmm2156\data\hackathon")
    DATA_PATH = Path(r"C:\Users\Luca\Desktop\Majeur-BA\RECOMMENDER-SYSTEM\mlsmm2156\data\small")
        
    # Content
    CONTENT_PATH = DATA_PATH / 'content'
    # - item
    ITEMS_FILENAME = 'movies.csv'
    ITEM_ID_COL = 'movieId'
    LABEL_COL = 'title'
    GENRES_COL = 'genres'
    POPULARITY_COL = 'popularity'

    # - Links
    LINKS_FILENAME = 'links.csv'
    IMDB_ID_COL = 'imdbId'
    TMDB_ID_COL = 'tmdbId'

    # - Tags
    TAGS_FILENAME = 'tags.csv'
    USER_ID_COL = 'userId'
    MOVIE_ID_COL = 'movieId'
    TAG_ID_COL = 'tag'
    TIMESTAMP_COL = 'timestamp'
    #ITEM_TAGS = [USER_ID_COL,MOVIE_ID_COL, TAG_ID_COL, TIMESTAMP_COL]
  
    # Evidence
    EVIDENCE_PATH = DATA_PATH / 'evidence'

    # - ratings
    RATINGS_FILENAME = 'ratings.csv'
    USER_ID_COL = 'userId'
    RATING_COL = 'rating'
    TIMESTAMP_COL = 'timestamp'
    USER_ITEM_RATINGS = [USER_ID_COL, ITEM_ID_COL, RATING_COL]

    # Rating scale
    # Rating scale
    RATINGS_SCALE = Reader(rating_scale=(0.5, 5))  # -- fill in here the ratings scale as a tuple (min_value, max_value)
    # Evaluation
    EVALUATION_PATH = Path(r"C:\Users\Luca\Desktop\Majeur-BA\RECOMMENDER-SYSTEM\mlsmm2156\evaluation")
    
    RATINGS_PATH = EVALUATION_PATH / 'ratings'
    