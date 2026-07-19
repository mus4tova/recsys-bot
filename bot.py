import os
import logging
import pandas as pd
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from search import load_data, find_matches, find_similar, imdb_url

load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')

logging.basicConfig(level=logging.INFO)

TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

# Load embeddings once at startup — keeps response times fast
movies, emb_desc, emb_genre, years, emb_director = load_data()


def _year(row) -> str:
    y = row.get('release_year', '')
    return str(y)[:4] if pd.notna(y) and str(y) != 'nan' else '?'


def _poster_url(row) -> str | None:
    path = row.get('poster_path')
    if pd.notna(path) and path:
        return f'{TMDB_IMAGE_BASE}{path}'
    return None


def _caption(row) -> str:
    """Build the photo caption for a single movie."""
    title   = row.get('title', '')
    year    = _year(row)
    genres  = row.get('genres', '')
    director = row.get('director', '')
    overview = str(row.get('overview', ''))[:300]  # keep caption short
    if overview and not overview.endswith('...'):
        overview += '...'

    lines = [f'🎬 *{title}* ({year})']
    if genres:
        lines.append(f'🎭 {genres}')
    if director and str(director) != 'nan':
        lines.append(f'🎥 {director}')
    if overview:
        lines.append(f'\n_{overview}_')
    return '\n'.join(lines)


def _carousel_keyboard(page: int, total: int, imdb_id) -> InlineKeyboardMarkup:
    """Navigation buttons: Prev / counter / Next + IMDb link."""
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton('← Prev', callback_data=f'page_{page - 1}'))
    nav.append(InlineKeyboardButton(f'{page + 1} / {total}', callback_data='noop'))
    if page < total - 1:
        nav.append(InlineKeyboardButton('Next →', callback_data=f'page_{page + 1}'))

    rows = [nav]
    link = imdb_url(imdb_id)
    if link:
        rows.append([InlineKeyboardButton('IMDb', url=link)])

    return InlineKeyboardMarkup(rows)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Hi! Send /similar <movie title> to find movies similar to it.\n'
        'Example: /similar Inception'
    )


async def similar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(context.args).strip()
    if not query:
        await update.message.reply_text('Usage: /similar <movie title>\nExample: /similar Inception')
        return

    matches = find_matches(movies, query)
    if matches.empty:
        await update.message.reply_text(f'No movies found for "{query}". Try a different title.')
        return

    keyboard = []
    for idx, row in matches.iterrows():
        label = f"{row['title']} ({_year(row)})"
        keyboard.append([InlineKeyboardButton(label, callback_data=f'movie_{idx}')])

    await update.message.reply_text(
        'Which movie did you mean?',
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def movie_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User picked a movie — compute similar films and show the first result."""
    query = update.callback_query
    await query.answer()

    idx = int(query.data.split('_')[1])
    results = find_similar(movies, emb_desc, emb_genre, years, emb_director, idx)

    # Store results in user session so carousel can navigate without recomputing
    result_indices = results.index.tolist()
    context.user_data['results'] = result_indices
    context.user_data['page']    = 0

    await _send_carousel(query, context, page=0)


async def page_turn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User pressed Prev / Next — update the carousel."""
    query = update.callback_query
    await query.answer()

    if query.data == 'noop':
        return

    page = int(query.data.split('_')[1])
    context.user_data['page'] = page
    await _send_carousel(query, context, page=page)


async def _send_carousel(query, context: ContextTypes.DEFAULT_TYPE, page: int):
    result_indices = context.user_data['results']
    total = len(result_indices)
    row = movies.loc[result_indices[page]]

    caption  = _caption(row)
    keyboard = _carousel_keyboard(page, total, row.get('imdbId'))
    poster   = _poster_url(row)

    if poster:
        # Delete the old message and send a new photo — Telegram can't edit photo messages
        await query.message.delete()
        await query.message.chat.send_photo(
            photo=poster,
            caption=caption,
            parse_mode='Markdown',
            reply_markup=keyboard,
        )
    else:
        await query.edit_message_text(
            text=caption,
            parse_mode='Markdown',
            reply_markup=keyboard,
        )


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('similar', similar))
    app.add_handler(CallbackQueryHandler(movie_selected, pattern=r'^movie_\d+$'))
    app.add_handler(CallbackQueryHandler(page_turn,      pattern=r'^(page_\d+|noop)$'))
    app.run_polling()


if __name__ == '__main__':
    main()
