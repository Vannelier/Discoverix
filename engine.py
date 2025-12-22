import csv
import os
import re
import unicodedata
from dataclasses import dataclass
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from wordfreq import zipf_frequency

# -------------------------
# Configuration
# -------------------------

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

REVEAL_EVERY = 10
MAX_REVEAL_STEPS = 6
SHOW_STEP0_AT_START = True

# CSV
CSV_ENCODINGS_TO_TRY = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']

# Scoring anchors
INCLUDE_ACCEPTED_IN_SCORING = True
INCLUDE_WORK_EMBEDDING_IN_SCORING = True

WEIGHT_CANONICAL = 1.0
WEIGHT_ACCEPTED = 0.98
WEIGHT_WORK_EMBEDDING = 0.90

# Dictionary gate (soft by default)
USE_DICTIONARY_GATE = True
MAX_TOKENS_DICTIONARY_GATE = 8
MIN_ZIPF_FR = 0.0
MIN_ZIPF_EN = 0.0
PROPER_NOUN_BYPASS_UPPERCASE = True

# Guess vector cache
GUESS_VEC_CACHE_MAX = 2048

# -------------------------
# Text utilities
# -------------------------


def fix_mojibake(s: str) -> str:
    # Best-effort fix for common UTF-8 decoded as cp1252/latin1.
    # If already correct, it should keep it unchanged.
    try:
        b = s.encode('latin-1', errors='strict')
        try:
            return b.decode('utf-8', errors='strict')
        except Exception:
            return s
    except Exception:
        return s


def normalize_basic(text: str) -> str:
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)

    text = fix_mojibake(text)

    # Normalize whitespace and punctuation variants
    text = (
        text.replace('\u00a0', ' ')
            .replace('\t', ' ')
            .replace('\r', ' ')
            .replace('\n', ' ')
            .replace('’', "'")
            .replace('‘', "'")
            .replace('“', '"')
            .replace('”', '"')
            .replace('…', ' ')
    )

    # Strip accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # Replace punctuation by spaces
    for ch in [
        "'", '"', ',', '.', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}',
        '-', '_', '/', '\\', '+', '#', '=', '&', '*', '@', '%', '^', '|', '~',
        '<', '>',
    ]:
        text = text.replace(ch, ' ')

    # Collapse spaces
    text = ' '.join(text.split())
    return text


def normalize_key(text: str) -> str:
    text = normalize_basic(text).lower().strip()
    return text


def split_pipe(s: str) -> list[str]:
    if not s:
        return []
    if not isinstance(s, str):
        s = str(s)
    parts = [p.strip() for p in s.split('|')]
    return [p for p in parts if p]


# -------------------------
# Puzzle loading and preparation
# -------------------------

_PUZZLE_INDEX_CACHE: dict = {'path': None, 'mtime': None, 'index': None}


def _get_puzzle_index(path: str) -> dict[str, dict]:
    """
    Charge puzzles.csv une seule fois et le met en cache.
    Rechargement automatique si le fichier est modifie (mtime).
    """
    path = path or 'puzzles.csv'
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None

    if (
        _PUZZLE_INDEX_CACHE.get('index') is not None
        and _PUZZLE_INDEX_CACHE.get('path') == path
        and _PUZZLE_INDEX_CACHE.get('mtime') == mtime
    ):
        return _PUZZLE_INDEX_CACHE['index']

    last_err = None
    for enc in CSV_ENCODINGS_TO_TRY:
        try:
            with open(path, 'r', encoding=enc, newline='') as f:
                reader = csv.DictReader(f)
                index: dict[str, dict] = {}
                for row in reader:
                    pid = (row.get('puzzle_id') or '').strip()
                    if not pid:
                        continue
                    index[pid] = row

            _PUZZLE_INDEX_CACHE['path'] = path
            _PUZZLE_INDEX_CACHE['mtime'] = mtime
            _PUZZLE_INDEX_CACHE['index'] = index
            return index
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    return {}


def load_puzzle(puzzle_id: str, path: str = 'puzzles.csv') -> dict | None:
    idx = _get_puzzle_index(path)
    row = idx.get(str(puzzle_id).strip())
    if not row:
        return None

    cleaned: dict = {}
    for k, v in row.items():
        if isinstance(v, str):
            cleaned[k] = fix_mojibake(v).strip()
        else:
            cleaned[k] = v
    return cleaned


def build_scoring_anchor_texts_and_weights(puzzle: dict) -> tuple[list[str], list[float]]:
    """
    Anchors used ONLY for scoring (0-99). This does NOT affect win condition.
    - canonical (weight 1.0)
    - accepted answers (near weight, e.g. 0.98)
    - work_embedding keywords (weight lower, e.g. 0.90) to improve 'associated' concepts
    """
    anchor_w: dict[str, float] = {}

    work_key = puzzle.get('work_canonical_normalized', '').strip()
    if not work_key:
        work_key = normalize_key(puzzle.get('work_canonical', ''))
    else:
        work_key = normalize_key(work_key)

    if work_key:
        anchor_w[work_key] = max(anchor_w.get(work_key, 0.0), WEIGHT_CANONICAL)

    if INCLUDE_ACCEPTED_IN_SCORING:
        for t in split_pipe(puzzle.get('accepted_answers', '')):
            t_key = normalize_key(t)
            if t_key:
                anchor_w[t_key] = max(anchor_w.get(t_key, 0.0), WEIGHT_ACCEPTED)

    if INCLUDE_WORK_EMBEDDING_IN_SCORING:
        for t in split_pipe(puzzle.get('work_embedding', '')):
            t_key = normalize_key(t)
            if t_key:
                anchor_w[t_key] = max(anchor_w.get(t_key, 0.0), WEIGHT_WORK_EMBEDDING)

    anchors = list(anchor_w.keys())
    weights = [float(anchor_w[a]) for a in anchors]
    return anchors, weights


def build_accepted_set_keys(puzzle: dict) -> set[str]:
    """
    Keys that trigger victory:
    - canonical
    - accepted_answers
    """
    s: set[str] = set()

    work_key = normalize_key(puzzle.get('work_canonical', ''))
    if work_key:
        s.add(work_key)

    for t in split_pipe(puzzle.get('accepted_answers', '')):
        t_key = normalize_key(t)
        if t_key:
            s.add(t_key)

    return s


# -------------------------
# Dictionary gate (soft)
# -------------------------

@lru_cache(maxsize=200000)
def token_exists_fr_or_en_alpha(token: str) -> bool:
    if not token:
        return False
    fr = zipf_frequency(token, 'fr')
    en = zipf_frequency(token, 'en')
    return (fr > MIN_ZIPF_FR) or (en > MIN_ZIPF_EN)


def is_roman(s: str) -> bool:
    return bool(re.fullmatch(r'[ivxlcdm]+', s))


def is_alnum_mixed(s: str) -> bool:
    return bool(re.search(r'[a-z]', s)) and bool(re.search(r'\d', s))


@lru_cache(maxsize=50000)
def guess_exists_in_fr_or_en(g_key: str) -> bool:
    """
    Hard gate is a bad fit for semantic games. This function is kept, but
    score_guess uses it in 'soft' mode: if false, we still accept the guess.
    """
    toks = g_key.split()
    if not toks:
        return False
    if len(toks) > MAX_TOKENS_DICTIONARY_GATE:
        return False

    for t in toks:
        if is_roman(t):
            continue
        if is_alnum_mixed(t):
            continue
        # Only check alpha-like tokens
        if re.fullmatch(r'[a-z]+', t):
            if not token_exists_fr_or_en_alpha(t):
                return False

    return True


# -------------------------
# Hint system
# -------------------------

def current_reveal_step(guess_count_unique: int) -> int:
    if SHOW_STEP0_AT_START:
        step = guess_count_unique // REVEAL_EVERY
    else:
        step = max(0, (guess_count_unique - 1) // REVEAL_EVERY)
    return int(max(0, min(MAX_REVEAL_STEPS - 1, step)))


def next_hint_in(guess_count_unique: int) -> int | None:
    step = current_reveal_step(guess_count_unique)
    if step >= MAX_REVEAL_STEPS - 1:
        return None
    mod = guess_count_unique % REVEAL_EVERY
    return int(REVEAL_EVERY - mod) if mod != 0 else int(REVEAL_EVERY)


def revealed_hints_list(reveal_steps: list[str], guess_count_unique: int) -> list[dict]:
    step = current_reveal_step(guess_count_unique)
    hints: list[dict] = []
    for i in range(0, min(step + 1, len(reveal_steps))):
        txt = reveal_steps[i] if i < len(reveal_steps) else ''
        if txt:
            hints.append({'step': i, 'text': txt})
    return hints


# -------------------------
# Scoring
# -------------------------

def score_from_similarity(sim: float) -> int:
    """
    Calibration for better gameplay:
    - early non-zero scores appear sooner
    - mid-range is spread (feels progressive)
    - 90+ remains rare but reachable near the target
    """
    sim = float(sim)
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0

    if sim <= 0.20:
        score = int(round((sim / 0.20) * 10.0))                 # 0..10
    elif sim <= 0.55:
        score = 10 + int(round(((sim - 0.20) / 0.35) * 60.0))   # 10..70
    elif sim <= 0.80:
        score = 70 + int(round(((sim - 0.55) / 0.25) * 23.0))   # 70..93
    else:
        score = 93 + int(round(((sim - 0.80) / 0.15) * 6.0))    # 93..99

    return max(0, min(99, score))


# -------------------------
# Model and caches
# -------------------------

@dataclass
class PreparedPuzzle:
    puzzle_id: str
    work_canonical: str
    character_name_full: str
    wikidata_qid: str
    accepted_set: set[str]
    anchors: list[str]
    v_anchors: np.ndarray
    anchor_weights: np.ndarray
    reveal_steps: list[str]


_model: SentenceTransformer | None = None
_prepared_cache: dict[str, PreparedPuzzle] = {}
_guess_vec_cache: OrderedDict[str, np.ndarray] = OrderedDict()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode_guess_vector(model: SentenceTransformer, g_key: str) -> np.ndarray:
    v = _guess_vec_cache.get(g_key)
    if v is not None:
        _guess_vec_cache.move_to_end(g_key, last=True)
        return v

    v = model.encode(
        [g_key],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    _guess_vec_cache[g_key] = v
    _guess_vec_cache.move_to_end(g_key, last=True)
    while len(_guess_vec_cache) > GUESS_VEC_CACHE_MAX:
        _guess_vec_cache.popitem(last=False)

    return v


def prepare_puzzle(puzzle_id: str, path: str = 'puzzles.csv') -> PreparedPuzzle:
    puzzle = load_puzzle(puzzle_id, path=path)
    if not puzzle:
        raise ValueError(f'puzzle introuvable: {puzzle_id}')

    work_canonical = (puzzle.get('work_canonical') or '').strip()
    character_name_full = (puzzle.get('character_name_full') or '').strip()
    wikidata_qid = (puzzle.get('wikidata_qid') or '').strip()

    accepted_set = build_accepted_set_keys(puzzle)

    anchors, weights = build_scoring_anchor_texts_and_weights(puzzle)
    if not anchors:
        anchors = [normalize_key(work_canonical)] if work_canonical else []
        weights = [WEIGHT_CANONICAL] * len(anchors)

    model = get_model()
    v_anchors = model.encode(
        anchors,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    anchor_weights = np.array(weights, dtype=np.float32)

    reveal_steps: list[str] = []
    for i in range(MAX_REVEAL_STEPS):
        k = f'reveal_step_{i}'
        v = puzzle.get(k, '')
        if isinstance(v, str):
            reveal_steps.append(v.strip())
        else:
            reveal_steps.append(str(v).strip() if v is not None else '')

    return PreparedPuzzle(
        puzzle_id=str(puzzle_id),
        work_canonical=work_canonical,
        character_name_full=character_name_full,
        wikidata_qid=wikidata_qid,
        accepted_set=accepted_set,
        anchors=anchors,
        v_anchors=v_anchors,
        anchor_weights=anchor_weights,
        reveal_steps=reveal_steps,
    )


def get_prepared(puzzle_id: str, path: str = 'puzzles.csv') -> PreparedPuzzle:
    p = _prepared_cache.get(puzzle_id)
    if p is not None:
        return p
    p = prepare_puzzle(puzzle_id, path=path)
    _prepared_cache[puzzle_id] = p
    return p


# -------------------------
# Main scoring entrypoint
# -------------------------

def score_guess(prepared: PreparedPuzzle, guess_raw: str) -> dict:
    g_key = normalize_key(guess_raw)
    if not g_key:
        return {'ok': False, 'error': 'empty'}

    # Strict win (before any gate)
    if g_key in prepared.accepted_set:
        return {
            'ok': True,
            'normalized': g_key,
            'is_win': True,
            'score': 100,
        }

    # Dictionary gate (soft): accept but tag and return low score
    if USE_DICTIONARY_GATE:
        bypass = False
        raw = (guess_raw or '')

        if PROPER_NOUN_BYPASS_UPPERCASE and any(c.isupper() for c in raw):
            bypass = True
        if not bypass and re.search(r'[0-9]', raw):
            bypass = True
        if not bypass and re.search(r'[+#=]', raw):
            bypass = True

        if not bypass and not guess_exists_in_fr_or_en(g_key):
            return {
                'ok': True,
                'normalized': g_key,
                'is_win': False,
                'score': 0,
                'note': 'not_in_dictionary',
            }

    # Embedding similarity to anchors
    model = get_model()
    v_guess = encode_guess_vector(model, g_key)

    # Cosine sims since embeddings are normalized
    sims = prepared.v_anchors @ v_guess  # shape: (n_anchors,)
    sims_weighted = sims * prepared.anchor_weights
    sim_max = float(np.max(sims_weighted)) if sims_weighted.size else 0.0

    score = score_from_similarity(sim_max)

    return {
        'ok': True,
        'normalized': g_key,
        'is_win': False,
        'score': int(score),
    }
