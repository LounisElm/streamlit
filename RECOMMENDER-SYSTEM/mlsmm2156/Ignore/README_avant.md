# 📊 Analytics Module — MLSMM2156 Recommender Systems

## Table of Contents
- [📊 Analytics Module — MLSMM2156 Recommender Systems](#-analytics-module--mlsmm2156-recommender-systems)
- [🧭 Objective](#-objective)
- [🧰 What’s inside](#-whats-inside)
- [🧪 How to use the notebook](#-how-to-use-the-notebook)
- [👨‍💻 To collaborators](#-to-collaborators)
- [📁 Dependencies](#-dependencies)

---

This notebook `analytics.ipynb` is the first module of the Movie Recommender System project for the course MLSMM2156 at UCLouvain.

It serves as a data exploration and diagnostic tool, giving a clear overview of the structure, content, and characteristics of the dataset used to train and evaluate recommendation models.

---

## 🧭 Objective
[(Back to top)](#table-of-contents)

The main purpose of this notebook is to analyze and summarize the MovieLens dataset in order to

- Understand the users, items (movies), and ratings distributions.
- Highlight important characteristics such as sparsity and long-tail distribution.

---

## 🧰 What’s inside
[(Back to top)](#table-of-contents)

The notebook is structured into the following steps

### 1. Data Loading
Loads movies, ratings, links, and tags using modular functions from the project (`loader.py`, `constants.py`).

### 2. Descriptive Statistics
Includes key insights such as

- Number of users, movies, and ratings.
- Distribution of ratings per movie.
- Range of movie release years (extracted from titles).
- Unique genres found in the dataset.
- Movies that were never rated.

### 3. Dataset Size Configuration
You can easily switch between dataset sizes (`tiny`, `small`, or `test`) by changing the data path in `constants.py`.

### 4. Long-Tail Property Visualization
Plots the distribution of the number of ratings per movie to illustrate the long-tail effect (few popular movies, many rarely rated ones).

### 5. Sparsity Analysis
- Computes the sparsity of the user-item rating matrix.
- Visualizes the sparsity using a 100×100 submatrix (users × movies) with `matplotlib.spy()`.

---

## 🧪 How to use the notebook
[(Back to top)](#table-of-contents)

1. Set up the Data

   - Download and unzip the dataset.

2. Switch dataset size
   - Open `constants.py` and modify the `DATA_PATH` to switch between datasets
     ```python
     DATA_PATH = Path('datatiny')  # or 'datasmall', 'datatest'
     ```

3. Run the notebook
   - Execute all cells in `analytics.ipynb` to perform the full analysis.


---

## 👨‍💻 To collaborators
[(Back to top)](#table-of-contents)

This notebook is meant to be executed before developing any recommendation algorithm.  
It ensures that the data is correctly loaded, and helps all members to understand
- The dataset's structure
- Data quality and completeness
- Key metrics (sparsity, rating distribution, etc.)

---

## 📁 Dependencies
[(Back to top)](#table-of-contents)

Make sure to install the following libraries

```bash
pip install pandas numpy matplotlib scipy
