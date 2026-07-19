import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'

# Scalar features use Gaussian similarity (value -> sigma); everything else
# is a vector feature compared with cosine similarity (dot product of
# pre-normalized embeddings).
SCALAR_SIGMAS = {
    'year': 10.0,
    'budget': 5.0,
    'popularity': 20.0,
}

_FILENAMES = {
    'description': 'emb_description.npy',
    'genre': 'emb_genre.npy',
    'director': 'emb_director.npy',
    'keywords': 'emb_keywords.npy',
    'cast': 'emb_cast.npy',
    'country': 'emb_country.npy',
    'year': 'emb_year.npy',
    'budget': 'emb_budget.npy',
    'popularity': 'emb_popularity.npy',
}

FEATURE_LABELS = {
    'description': 'Description',
    'genre': 'Genre',
    'director': 'Director',
    'keywords': 'Keywords',
    'cast': 'Cast',
    'country': 'Country',
    'year': 'Year',
    'budget': 'Budget',
    'popularity': 'Popularity',
}

# Weights are percentages (0-100) that add up to 100; the absolute scale
# doesn't affect ranking (only relative proportions matter), it's just a
# more intuitive unit for the /similar weights UI.
DEFAULT_WEIGHTS = {
    'description': 40,
    'genre': 20,
    'year': 10,
    'director': 10,
    'keywords': 10,
    'cast': 5,
    'country': 5,
    'budget': 0,
    'popularity': 0,
}


def load_data():
    movies = pd.read_csv(DATA_DIR / 'movies_clean.csv')
    features = {name: np.load(DATA_DIR / fname) for name, fname in _FILENAMES.items()}

    # Merge imdbId from links.csv so we can build IMDB links in the bot
    links = pd.read_csv(DATA_DIR / 'links.csv', dtype={'imdbId': str})
    movies = movies.merge(links[['movieId', 'imdbId']], on='movieId', how='left')

    return movies, features


def imdb_url(imdb_id) -> str:
    if pd.isna(imdb_id):
        return ''
    return f'https://www.imdb.com/title/tt{str(imdb_id).zfill(7)}/'


def find_matches(movies: pd.DataFrame, query: str, n: int = 5) -> pd.DataFrame:
    """Return up to n movies whose title contains the query string."""
    mask = movies['title'].str.lower().str.contains(query.lower(), na=False)
    return movies[mask].head(n)


def find_similar(
    movies: pd.DataFrame,
    features: dict,
    idx: int,
    weights: dict = None,
    year_range: tuple = None,
    k: int = 10,
) -> pd.DataFrame:
    """
    Return top-k movies most similar to movies.iloc[idx].
    `weights` maps feature name -> weight (see DEFAULT_WEIGHTS for the
    available names). Vector features are compared with cosine similarity,
    scalar features (year, budget, popularity) with Gaussian similarity.
    `year_range`, if given, is a (min_year, max_year) hard filter: movies
    outside it are excluded from the candidates entirely, regardless of
    weights.
    """
    weights = weights or DEFAULT_WEIGHTS
    scores = np.zeros(len(movies), dtype=np.float32)

    for name, w in weights.items():
        if w == 0:
            continue
        emb = features[name]
        if name in SCALAR_SIGMAS:
            sigma = SCALAR_SIGMAS[name]
            scores += w * np.exp(-((emb - emb[idx]) ** 2) / (2 * sigma ** 2))
        else:
            scores += w * (emb @ emb[idx])

    if year_range is not None:
        lo, hi = year_range
        out_of_range = (features['year'] < lo) | (features['year'] > hi)
        scores[out_of_range] = -1

    scores[idx] = -1  # exclude the query movie itself
    top_k = np.argsort(scores)[::-1][:k]
    return movies.iloc[top_k]
