import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


def load_data():
    movies      = pd.read_csv(DATA_DIR / 'movies_clean.csv')
    emb_desc    = np.load(DATA_DIR / 'emb_description.npy')
    emb_genre   = np.load(DATA_DIR / 'emb_genre.npy')
    years       = np.load(DATA_DIR / 'emb_year.npy')
    emb_director = np.load(DATA_DIR / 'emb_director.npy')

    # Merge imdbId from links.csv so we can build IMDB links in the bot
    links = pd.read_csv(DATA_DIR / 'links.csv', dtype={'imdbId': str})
    movies = movies.merge(links[['movieId', 'imdbId']], on='movieId', how='left')

    return movies, emb_desc, emb_genre, years, emb_director


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
    emb_desc: np.ndarray,
    emb_genre: np.ndarray,
    years: np.ndarray,
    emb_director: np.ndarray,
    idx: int,
    w_desc: float = 0.4,
    w_genre: float = 0.3,
    w_year: float = 0.2,
    w_director: float = 0.1,
    year_sigma: float = 10.0,
    k: int = 10,
) -> pd.DataFrame:
    """
    Return top-k movies most similar to movies.iloc[idx].
    Similarity is a weighted sum of four independent signals:
      - description cosine similarity (MiniLM embeddings)
      - genre cosine similarity (multi-hot)
      - year Gaussian similarity (sigma in years)
      - director cosine similarity (genre profile)
    """
    year_sim = np.exp(-((years - years[idx]) ** 2) / (2 * year_sigma ** 2))

    scores = (
        w_desc     * (emb_desc     @ emb_desc[idx])     +
        w_genre    * (emb_genre    @ emb_genre[idx])    +
        w_year     * year_sim                           +
        w_director * (emb_director @ emb_director[idx])
    )
    scores[idx] = -1  # exclude the query movie itself

    top_k = np.argsort(scores)[::-1][:k]
    return movies.iloc[top_k]
