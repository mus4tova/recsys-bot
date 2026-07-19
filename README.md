# Movie Recommender Bot

A Telegram bot with two independent movie-discovery modes: content-based
"similar movies" search, and (planned) personalized recommendations from
user ratings.

## Option 1 — Similar movies (`/similar <title>`)

User enters a title, picks the right match from a list of candidates, and
the bot returns the 10 most similar movies as a swipeable carousel (poster,
genres, director, overview, IMDb link).

Similarity is a weighted sum of independent per-feature signals:

```
score = w_desc * sim_desc + w_genre * sim_genre + w_year * sim_year + w_director * sim_dir + ...
```

| Feature | Method | Dim |
|---|---|---|
| Description | MiniLM (`all-MiniLM-L6-v2`) on title+genres+overview | 384 |
| Genre | multi-hot over ~20 genres | ~20 |
| Year | Gaussian similarity on release year | scalar |
| Director | mean genre profile across their filmography | ~20 |
| Keywords | multi-hot over top-500 TMDB keywords | 500 |
| Cast | mean genre profile across an actor's filmography | ~20 |
| Country | multi-hot over production countries | varies |
| Budget | Gaussian similarity on log(budget) | scalar |
| Popularity | Gaussian similarity on TMDB popularity score | scalar |

**Status:** the notebook (`2_build_embeddings.ipynb`) computes and validates
all 9 signals above with a configurable-weight search function. The
production path (`search.py`, used by `bot.py`) only wires up the original
4 (description, genre, year, director) — keywords/cast/country/budget/
popularity are computed to disk but not yet used by the bot.

## Option 2 — Personalized recommendations (`/rate`, `/recommend`)

User rates movies, bot suggests what to watch next. Planned approach:
Matrix Factorization (SVD) trained on MovieLens 25M ratings; user vector =
weighted average of item embeddings for rated movies; nearest neighbors via
FAISS.

**Status:** not started. No SVD/FAISS code exists yet, and there is no
ratings storage for users of the bot.

## Current project status

- [x] Data prep — `movies_clean.csv` merges MovieLens + TMDB (director,
      cast, keywords, budget, popularity, etc.)
- [x] Content embeddings — all 9 signals above built and saved to `data/*.npy`
      (`2_build_embeddings.ipynb`)
- [ ] Wire the full 9-signal weighting into `search.py` (currently only 4
      of the 9 are used in production)
- [x] Telegram bot skeleton (`bot.py`) — `/start`, `/similar`, title
      disambiguation, paginated result carousel, TMDB posters, IMDb links
- [ ] **Dependencies not installed/pinned** — `python-telegram-bot` and
      `python-dotenv` are missing from `.venv`, `bot.py` currently fails at
      import (`ModuleNotFoundError: dotenv`). No `requirements.txt` /
      `pyproject.toml` exists.
- [ ] SQLite persistence for user ratings (needed for Option 2)
- [ ] SVD training + FAISS index for Option 2

Two stray files, `data/item_embeddings_content.npy` and
`data/movies_content.index`, predate the rest of the embeddings and aren't
read by any current code — looks like an abandoned early FAISS experiment.

## Stack

sentence-transformers · numpy · pandas · FAISS · SVD (planned:
surprise/scipy) · python-telegram-bot · SQLite (planned)

## Datasets

### MovieLens 25M

**Goal:** supplies the user-rating signal needed for collaborative
filtering (Option 2) and the base movie/genre catalog everything else is
built on.

25,000,095 ratings and 1,093,360 tag applications across 62,423 movies,
from 162,541 users (Jan 1995 – Nov 2019), published by GroupLens
(movielens.org). Key files used here:

- `ratings.csv` — `userId, movieId, rating (0.5–5 stars), timestamp`
- `movies.csv` — `movieId, title, genres` (pipe-separated genre list)
- `links.csv` — `movieId, imdbId, tmdbId`, used to join in TMDB metadata
  and to build IMDb links in the bot
- `tags.csv`, `genome-scores.csv`, `genome-tags.csv` — free-text tags and
  the tag-genome relevance matrix (not currently used, potential future
  content signal)

Full details in `data/README.txt`.

### TMDB API

**Goal:** enriches the bare MovieLens catalog with the descriptive
metadata the content-based embeddings in Option 1 are built from —
MovieLens alone only has title, year, and genres.

Fetched per movie (via `links.csv` → `tmdbId`) and merged into
`movies_clean.csv`: overview/tagline, director/writer/composer/
cinematographer/producer, cast, keywords, budget, revenue, popularity,
vote average/count, poster/backdrop paths, origin country, spoken
languages, and collection/production company info. Results are cached in
`data/tmdb_cache.csv` to avoid re-hitting the API.
