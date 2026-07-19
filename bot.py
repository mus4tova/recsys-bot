import os
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ForceReply
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from search import (
    load_data,
    find_matches,
    find_similar,
    imdb_url,
    DEFAULT_WEIGHTS,
    FEATURE_LABELS,
)

load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')

logging.basicConfig(level=logging.INFO)

TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

WEIGHT_STEP = 5
YEAR_STEP = 5
CRITERIA_ORDER = list(DEFAULT_WEIGHTS.keys())

# Load embeddings once at startup — keeps response times fast
movies, features = load_data()

YEAR_MIN = int(movies['release_year'].dropna().min())
YEAR_MAX = int(movies['release_year'].dropna().max())


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


def _year_keyboard(year_range: list) -> InlineKeyboardMarkup:
    """Build the year-range step: -/value/+ for min and max, plus Next."""
    lo, hi = year_range
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton('➖', callback_data='y_dec_min'),
            InlineKeyboardButton(f'From: {lo}', callback_data='noop'),
            InlineKeyboardButton('➕', callback_data='y_inc_min'),
        ],
        [
            InlineKeyboardButton('➖', callback_data='y_dec_max'),
            InlineKeyboardButton(f'To: {hi}', callback_data='noop'),
            InlineKeyboardButton('➕', callback_data='y_inc_max'),
        ],
        [InlineKeyboardButton('✏️ Enter manually', callback_data='y_manual')],
        [InlineKeyboardButton('Next → Priorities', callback_data='y_next')],
    ])


def _weights_keyboard(weights: dict) -> InlineKeyboardMarkup:
    """Build the per-search weights menu: one -/value/+ row per criterion,
    plus a button to run the search with the current weights."""
    rows = []
    for name in CRITERIA_ORDER:
        label = f'{FEATURE_LABELS[name]}: {weights[name]}%'
        rows.append([
            InlineKeyboardButton('➖', callback_data=f'w_dec_{name}'),
            InlineKeyboardButton(label, callback_data='noop'),
            InlineKeyboardButton('➕', callback_data=f'w_inc_{name}'),
        ])
    rows.append([InlineKeyboardButton('🔍 Find similar', callback_data='w_search')])
    return InlineKeyboardMarkup(rows)


def _weights_prompt_text(weights: dict) -> str:
    used = sum(weights.values())
    remaining = 100 - used
    return (
        'Set how much each criterion should matter, then tap Find similar.\n\n'
        f'Allocated: {used}% · Remaining: {remaining}%'
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Hi! Send /similar <movie title> to find movies similar to it.\n'
        'Example: /similar Inception\n\n'
        'After you pick a movie, you can narrow down the release year range '
        'and set how much each criterion (genre, cast...) matters before '
        'running the search.'
    )


async def _show_weights_step(query, context: ContextTypes.DEFAULT_TYPE):
    weights = context.user_data.setdefault('weights', dict(DEFAULT_WEIGHTS))
    await query.edit_message_text(
        text=_weights_prompt_text(weights),
        reply_markup=_weights_keyboard(weights),
    )


async def year_range_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User tapped -/+, 'Enter manually', or 'Next' in the year-range step."""
    query = update.callback_query
    await query.answer()

    if query.data == 'y_manual':
        context.user_data['awaiting_year_input'] = True
        await query.message.reply_text(
            f'Send the year range as `from-to`, e.g. `1990-2010` '
            f'(dataset covers {YEAR_MIN}-{YEAR_MAX}).',
            parse_mode='Markdown',
            reply_markup=ForceReply(selective=True, input_field_placeholder='e.g. 1990-2010'),
        )
        return

    context.user_data['awaiting_year_input'] = False

    if query.data == 'y_next':
        await _show_weights_step(query, context)
        return

    year_range = context.user_data.setdefault('year_range', [YEAR_MIN, YEAR_MAX])
    _, direction, bound = query.data.split('_')
    delta = YEAR_STEP if direction == 'inc' else -YEAR_STEP
    i = 0 if bound == 'min' else 1

    year_range[i] = max(YEAR_MIN, min(YEAR_MAX, year_range[i] + delta))
    if year_range[0] > year_range[1]:
        year_range[1 - i] = year_range[i]  # don't let the bounds cross

    await query.edit_message_reply_markup(reply_markup=_year_keyboard(year_range))


async def year_range_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User typed a year range in reply to the 'Enter manually' prompt."""
    if not context.user_data.get('awaiting_year_input'):
        return  # not expecting free text right now — ignore

    years = re.findall(r'\d+', update.message.text)
    if len(years) < 2:
        await update.message.reply_text('Please send two years, e.g. `1990-2010`.', parse_mode='Markdown')
        return

    lo, hi = sorted(int(y) for y in years[:2])
    lo = max(YEAR_MIN, min(YEAR_MAX, lo))
    hi = max(YEAR_MIN, min(YEAR_MAX, hi))

    context.user_data['year_range'] = [lo, hi]
    context.user_data['awaiting_year_input'] = False

    await update.message.reply_text(
        'Narrow down the release year range (optional), then tap Next.',
        reply_markup=_year_keyboard([lo, hi]),
    )


async def weight_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User tapped -/+ to adjust a weight, or 'Find similar' to run the search."""
    query = update.callback_query
    await query.answer()

    weights = context.user_data.setdefault('weights', dict(DEFAULT_WEIGHTS))

    if query.data == 'w_search':
        idx = context.user_data.get('pending_idx')
        if idx is None:
            return
        year_range = context.user_data.get('year_range')
        results = find_similar(
            movies, features, idx,
            weights=weights,
            year_range=tuple(year_range) if year_range else None,
        )

        # Store results in user session so carousel can navigate without recomputing
        context.user_data['results'] = results.index.tolist()
        context.user_data['page']    = 0
        await _send_carousel(query, context, page=0)
        return

    direction, name = query.data.split('_', 2)[1:]
    if direction == 'inc':
        remaining = 100 - sum(weights.values())
        delta = max(0, min(WEIGHT_STEP, remaining))  # never allocate past 100% total
    else:
        delta = -WEIGHT_STEP
    weights[name] = max(0, min(100, weights[name] + delta))

    await query.edit_message_text(
        text=_weights_prompt_text(weights),
        reply_markup=_weights_keyboard(weights),
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
    """User picked a movie — let them narrow the year range before weights."""
    query = update.callback_query
    await query.answer()

    idx = int(query.data.split('_')[1])
    context.user_data['pending_idx'] = idx
    context.user_data['awaiting_year_input'] = False
    year_range = context.user_data.setdefault('year_range', [YEAR_MIN, YEAR_MAX])

    await query.edit_message_text(
        text='Narrow down the release year range (optional), then tap Next.',
        reply_markup=_year_keyboard(year_range),
    )


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
    app.add_handler(CallbackQueryHandler(movie_selected,   pattern=r'^movie_\d+$'))
    app.add_handler(CallbackQueryHandler(page_turn,        pattern=r'^(page_\d+|noop)$'))
    app.add_handler(CallbackQueryHandler(year_range_button, pattern=r'^(y_(inc|dec)_(min|max)|y_manual|y_next)$'))
    app.add_handler(CallbackQueryHandler(weight_button,    pattern=r'^(w_(inc|dec)_\w+|w_search)$'))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, year_range_text))
    app.run_polling()


if __name__ == '__main__':
    main()
