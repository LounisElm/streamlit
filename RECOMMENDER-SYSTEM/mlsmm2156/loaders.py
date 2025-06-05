# third parties imports
import pandas as pd
from surprise import Dataset, SVD, Reader, accuracy

# local imports
from constants import Constant as C
import time



def load_ratings(surprise_format=False, hack=False):
    if hack:
        evidence_path = C.HACK_PATH / 'evidence'
    else:
        evidence_path = C.EVIDENCE_PATH
    df_ratings = pd.read_csv(evidence_path / C.RATINGS_FILENAME)
    if surprise_format:
        data_ratings = Dataset.load_from_df(df_ratings[C.USER_ITEM_RATINGS],C.RATINGS_SCALE)
        return data_ratings
    else:
        return df_ratings


def load_items():
    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)
    return df_items

def load_links():
    df_links = pd.read_csv(C.CONTENT_PATH / C.LINKS_FILENAME)
    return df_links

def load_tags():
    df_tags = pd.read_csv(C.CONTENT_PATH / C.TAGS_FILENAME)
    return df_tags


def export_evaluation_report(evalution_report):
    """
    Exports the evaluation report DataFrame to a CSV file named with the current date.
    """

    # Format today's date for the filename
    filename = time.strftime("%Y_%m_%d_%H_%M_%S") + "_report.csv"
    filepath = C.EVALUATION_PATH / filename

    # Export the DataFrame to CSV
    evalution_report.to_csv(filepath, index=False)
    print(f"Evaluation report exported to: {filepath}")


def load_genome_data():
    """
    Charge les données du dataset Genome
    """
    try:
        # Charger les scores de pertinence des tags
        hack = C.HACK_PATH / 'content' if C.HACK_PATH.exists() else C.CONTENT_PATH
        C.CONTENT_PATH = hack
        genome_scores = pd.read_csv(C.CONTENT_PATH / "genome-scores.csv")
        # Charger les tags
        genome_tags = pd.read_csv(C.CONTENT_PATH / 'genome-tags.csv')
        return genome_scores, genome_tags
    except FileNotFoundError:
        print("Warning: Genome dataset not found. Tag-based features will be disabled.")
        return None, None

