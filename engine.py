"""Discoverix Game Engine"""
import os
import re
import csv
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict
from threading import Lock
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

PUZZLES_DIR = os.environ.get("PUZZLES_DIR", os.path.join(BASE_DIR, "puzzles")
MODEL_NAME = os.getenv('MODEL_NAME', 'intfloat/multilingual-e5-small')
EMBED_ROLE_GUESS = os.getenv('EMBED_ROLE_GUESS', 'query')
EMBED_ROLE_ANCHOR = os.getenv('EMBED_ROLE_ANCHOR', 'passage')
ENCODINGS_TO_TRY = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']
WEIGHT_CANONICAL = 1.0
WEIGHT_WORK_EMBEDDING = 1.0
WEIGHT_GLOBAL_CONTEXT = 0.90
SIM_GLACIAL_MAX = 0.80
SIM_EPS = 1e-6
USE_DICTIONARY_GATE = str(os.getenv('USE_DICTIONARY_GATE', 'false')).lower() == 'true'
PROPER_NOUN_BYPASS_UPPERCASE = True
GUESS_VEC_CACHE_MAX = int(os.getenv('GUESS_VEC_CACHE_MAX', '2048'))

NEW_CANONICAL_COL = 'label_display'
NEW_WIKIDATA_URL_COL = 'wikidata_url'
NEW_EMBED_COLS = ['instance_of', 'part_of', 'subclass_of', 'country_of_origin', 'wikipedia_tags']
NEW_HINT_COLS = ['Category', 'instance_of', 'country_of_origin', 'description_en']
GLOBAL_CONTEXT_COLS = ['description_en', *NEW_EMBED_COLS]
MAX_NEW_EMBED_TERMS = 48
EXACT_ANCHOR_MIN_SCORE = 75

def fix_mojibake(s: str) -> str:
    if not s:
        return s
    for enc in ENCODINGS_TO_TRY:
        try:
            b = s.encode(enc, errors='ignore')
            out = b.decode(enc, errors='ignore')
            if out:
                return out
        except Exception:
            pass
    return s


def normalize_text(s: str) -> str:
    if not s:
        return ''
    s = fix_mojibake(str(s)).strip().lower()
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))
    return re.sub(r'[\s\-_]+', ' ', s).strip()


def split_multi_value(raw: str) -> list:
    if not raw:
        return []
    s = str(raw).strip()
    sep = '|' if '|' in s else '\n' if '\n' in s else ';' if ';' in s else None
    return [p.strip() for p in (s.split(sep) if sep else [s]) if p.strip()]

_CAP_FIRST_RE = re.compile(r"^(\s*)([A-Za-zÀ-ÖØ-öø-ÿ])", re.UNICODE)

def capitalize_first_word(text: str) -> str:
    return _CAP_FIRST_RE.sub(lambda m: m.group(1) + m.group(2).upper(), str(text).strip(), count=1) if text else ''


def _cell(row: dict, key: str) -> str:
    v = row.get(key) if row and key else None
    return str(v).strip() if v and str(v).strip() not in ('', 'nan') else ''

_WIKIDATA_QID_RE = re.compile(r'(Q\d+)')

def parse_wikidata_qid_from_url(url: str) -> str:
    return m.group(1) if (m := _WIKIDATA_QID_RE.search(str(url))) else '' if url else ''


def build_global_context_text(row: dict) -> str:
    parts, seen = [], set()
    for col in GLOBAL_CONTEXT_COLS:
        if raw := _cell(row, col):
            for item in split_multi_value(raw):
                if (n := normalize_text(item)) and n not in seen:
                    seen.add(n)
                    parts.append(item.strip())
    return ' | '.join(parts)


def build_new_work_embedding(row: dict, canonical_norm: str) -> list:
    seen, out = set(), []
    for col in NEW_EMBED_COLS:
        if raw := _cell(row, col):
            for item in split_multi_value(raw):
                if (n := normalize_text(item)) and n != canonical_norm and n not in seen:
                    seen.add(n)
                    out.append(item.strip())
                    if len(out) >= MAX_NEW_EMBED_TERMS:
                        return out
    return out


def build_new_hints(row: dict) -> list:
    """Build hints from row data. Prioritizes Hint1-HintN columns if present."""
    out, seen = [], set()

    # First, check for new structured hint columns (Hint1, Hint2, etc.)
    hint_cols = [col for col in row.keys() if col.startswith('Hint') and col[4:].isdigit()]
    hint_cols_sorted = sorted(hint_cols, key=lambda x: int(x[4:]))

    for col in hint_cols_sorted:
        if raw := _cell(row, col):
            # Parse TYPE:VALUE format
            if ':' in raw:
                parts = raw.split(':', 1)
                if len(parts) == 2:
                    hint_type = parts[0].strip()
                    hint_value = parts[1].strip()
                    if hint_value and (n := normalize_text(hint_value)) and n not in seen:
                        seen.add(n)
                        out.append({'type': hint_type, 'text': hint_value})
                        if len(out) >= 64:
                            return out

    # Fallback to legacy hint extraction if no Hint columns found
    if not out:
        for col in NEW_HINT_COLS:
            if raw := _cell(row, col):
                for item in split_multi_value(raw):
                    text = capitalize_first_word(item)
                    if (n := normalize_text(text)) and n not in seen:
                        seen.add(n)
                        out.append({'type': col, 'text': text})
                        if len(out) >= 64:
                            return out

    return out


def _should_bypass_dictionary_gate(raw_guess: str) -> bool:
    if not PROPER_NOUN_BYPASS_UPPERCASE or not raw_guess:
        return False
    s = str(raw_guess).strip()
    if s.isupper() and len(s) >= 2:
        return True
    if any(ch.isdigit() for ch in s):
        return True
    tokens = [t for t in re.split(r'\s+', s) if t]
    return len(tokens) >= 2 and sum(1 for t in tokens if t and t[0].isalpha() and t[0].isupper()) >= 2

def guess_exists_in_dictionary(guess_raw: str) -> bool:
    # Clarified logic:
    # - if no input, return False
    # - bypass gate for proper nouns / uppercase tokens
    # - otherwise consider it present if it contains at least 2 alphabetic characters
    if not guess_raw:
        return False
    if _should_bypass_dictionary_gate(guess_raw):
        return True
    return sum(1 for c in guess_raw if c.isalpha()) >= 2


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: int
    character_name_full: str
    work_canonical: str
    accepted_answers: list
    work_embedding: list
    reveal_steps: list
    wikidata_qid: str
    wikidata_url: str
    global_context: str

@dataclass(frozen=True)
class PreparedPuzzle:
    puzzle: Puzzle
    canonical_norm: str
    accepted_norm_set: set
    reveal_steps: list
    anchors_text: list
    anchors_weight: list
    anchors_norm: list
    anchors_kind: list
    anchor_vecs: list

def _read_csv_rows(path: str) -> list:
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            with open(path, 'r', encoding=enc, newline='') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f'Unable to read CSV: {path} ({last_err})')


def get_available_campaigns() -> dict:
    """
    Retourne la liste des campagnes disponibles
    Format: {campaign_name: filepath}
    """
    campaigns = {}

    if not os.path.exists(PUZZLES_DIR):
        logger = logging.getLogger("discoverix.engine")
        logger.warning("Puzzles directory not found: %s", PUZZLES_DIR)
        return campaigns

    csv_files = [f for f in os.listdir(PUZZLES_DIR) if f.endswith('.csv')]

    for csv_file in csv_files:
        # Format: campaign_CATEGORYNAME_timestamp.csv ou puzzle_timestamp.csv (legacy)
        if csv_file.startswith('campaign_'):
            parts = csv_file.replace('.csv', '').split('_')
            if len(parts) >= 2:
                # campaign_CATEGORYNAME_timestamp -> CATEGORYNAME
                campaign_name = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                campaigns[campaign_name] = os.path.join(PUZZLES_DIR, csv_file)
        else:
            # Legacy format: puzzle_timestamp.csv -> "Default"
            campaigns['Default'] = os.path.join(PUZZLES_DIR, csv_file)

    return campaigns


@lru_cache(maxsize=10)
def load_puzzles(campaign_name: str = None) -> list:
    """
    Charge les puzzles d'une campagne spécifique
    Si campaign_name est None, charge la première campagne disponible
    """
    puzzles: list = []

    campaigns = get_available_campaigns()

    if not campaigns:
        logger = logging.getLogger("discoverix.engine")
        logger.warning("No campaigns found in %s", PUZZLES_DIR)
        return puzzles

    # Si aucune campagne spécifiée, prendre la première
    if campaign_name is None:
        campaign_name = list(campaigns.keys())[0]

    if campaign_name not in campaigns:
        logger = logging.getLogger("discoverix.engine")
        logger.warning("Campaign '%s' not found", campaign_name)
        return puzzles

    csv_path = campaigns[campaign_name]

    try:
        rows = _read_csv_rows(csv_path)

        for r in rows:
            try:
                pid = int(_cell(r, 'puzzle_id'))
            except Exception:
                continue

            # Canonical answer
            work = _cell(r, NEW_CANONICAL_COL) or _cell(r, 'work_canonical')
            if not work:
                continue
            canonical_norm = normalize_text(work)

            # Character name
            character = _cell(r, 'character_name_full') or _cell(r, 'qid')

            # Accepted answers
            accepted_raw = _cell(r, 'accepted_answers')
            accepted = [a.strip() for a in accepted_raw.split('|') if a.strip()] if accepted_raw else []

            # Work embedding
            work_emb_raw = _cell(r, 'work_embedding')
            if work_emb_raw:
                work_emb = [a.strip() for a in work_emb_raw.split('|') if a.strip()]
            else:
                work_emb = build_new_work_embedding(r, canonical_norm=canonical_norm)

            # Hints
            reveal_steps = build_new_hints(r)

            # Global context
            global_context = build_global_context_text(r)
            if not global_context:
                global_context = ' | '.join([x for x in work_emb if x.strip()])

            # Wikidata
            wikidata_url = _cell(r, NEW_WIKIDATA_URL_COL)
            qid = _cell(r, 'qid') or parse_wikidata_qid_from_url(wikidata_url)
            if not wikidata_url and qid:
                wikidata_url = f'https://www.wikidata.org/wiki/{qid}'

            puzzles.append(
                Puzzle(
                    puzzle_id=pid,
                    character_name_full=character,
                    work_canonical=work,
                    accepted_answers=accepted,
                    work_embedding=work_emb,
                    reveal_steps=reveal_steps,
                    wikidata_qid=qid,
                    wikidata_url=wikidata_url,
                    global_context=global_context,
                )
            )

    except Exception as e:
        logger = logging.getLogger("discoverix.engine")
        logger.exception("Error loading campaign '%s'", campaign_name)

    puzzles.sort(key=lambda p: p.puzzle_id)
    logger = logging.getLogger("discoverix.engine")
    logger.info("Loaded %d puzzles", len(puzzles))
    return puzzles


@lru_cache(maxsize=10)
def get_puzzle_ids_for_campaign(campaign_name: str = None) -> List[int]:
    """
    Retourne la liste ordonnée des IDs de puzzle pour une campagne.
    """
    puzzles = load_puzzles(campaign_name)
    # Les puzzles sont déjà triés par puzzle_id dans load_puzzles
    return [p.puzzle_id for p in puzzles]


@lru_cache(maxsize=4096)
def prepare_puzzle(puzzle_id: int, campaign_name: str = None) -> PreparedPuzzle:
    """Prépare un puzzle avec ancres et vecteurs"""
    puzzles = load_puzzles(campaign_name)
    p = next((x for x in puzzles if x.puzzle_id == puzzle_id), None)
    if p is None:
        raise KeyError(f'Puzzle id not found: {puzzle_id} in campaign {campaign_name}')

    canonical_norm = normalize_text(p.work_canonical)
    accepted_norm = [normalize_text(a) for a in p.accepted_answers]
    accepted_norm_set = set([a for a in accepted_norm if a])

    anchors_text: list = []
    anchors_weight: list = []
    anchors_norm: list = []
    anchors_kind: list = []
    anchor_vecs: list = []

    # Anchor[0]: canonical
    anchors_text.append(p.work_canonical)
    anchors_weight.append(WEIGHT_CANONICAL)
    anchors_norm.append(canonical_norm)
    anchors_kind.append('canonical')

    # Tokens du canonical
    stop = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'd', 'l', 'au', 'aux',
        'et', 'ou', 'a', 'à', 'en', 'sur', 'dans', 'pour', 'par', 'avec',
    }
    for tok in [t for t in canonical_norm.split(' ') if t]:
        if len(tok) < 3:
            continue
        if tok in stop:
            continue
        if tok in anchors_norm:
            continue
        anchors_text.append(tok)
        anchors_weight.append(WEIGHT_CANONICAL)
        anchors_norm.append(tok)
        anchors_kind.append('canon_token')

    # Work embedding
    for w in p.work_embedding:
        ww = str(w or '').strip()
        if not ww:
            continue
        wn = normalize_text(ww)
        if not wn:
            continue
        if wn in anchors_norm:
            continue
        anchors_text.append(ww)
        anchors_weight.append(WEIGHT_WORK_EMBEDDING)
        anchors_norm.append(wn)
        anchors_kind.append('work_embedding')

    # Precalcul vecteurs
    for a_text in anchors_text:
        try:
            vec = embed_text(a_text, role=EMBED_ROLE_ANCHOR)
        except Exception:
            vec = np.zeros((get_model().get_sentence_embedding_dimension(),), dtype=np.float32)
        anchor_vecs.append(vec)

    return PreparedPuzzle(
        puzzle=p,
        canonical_norm=canonical_norm,
        accepted_norm_set=accepted_norm_set,
        reveal_steps=p.reveal_steps,
        anchors_text=anchors_text,
        anchors_weight=anchors_weight,
        anchors_norm=anchors_norm,
        anchors_kind=anchors_kind,
        anchor_vecs=anchor_vecs,
    )


# -------------------------
# Model + Embedding cache
# -------------------------

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    logger = logging.getLogger("discoverix.engine")
    logger.info("Loading embedding model: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


class LRUCache:
    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self._od = OrderedDict()
        self._lock = Lock()

    def get(self, k):
        with self._lock:
            if k in self._od:
                v = self._od.pop(k)
                self._od[k] = v
                return v
            return None

    def set(self, k, v):
        with self._lock:
            if k in self._od:
                self._od.pop(k)
            self._od[k] = v
            if len(self._od) > self.max_size:
                self._od.popitem(last=False)


_guess_vec_cache = LRUCache(GUESS_VEC_CACHE_MAX)


def _is_e5_model(name: str) -> bool:
    return 'e5' in (name or '').lower()


def _e5_prefix(role: str, text: str) -> str:
    role = (role or '').strip().lower()
    if role not in {'query', 'passage'}:
        role = 'query'
    return f'{role}: {text}'


def embed_text(s: str, role: str = 'query') -> np.ndarray:
    """Retourne un embedding L2-normalisé"""
    s = str(s)
    cache_key = f'{role}\n{s}'
    cached = _guess_vec_cache.get(cache_key)
    if cached is not None:
        return cached

    model = get_model()
    t = s.strip()
    if not t:
        v = np.zeros((model.get_sentence_embedding_dimension(),), dtype=np.float32)
        _guess_vec_cache.set(cache_key, v)
        return v

    if _is_e5_model(MODEL_NAME):
        inp = _e5_prefix(role, t)
        v = model.encode([inp], normalize_embeddings=True)[0]
    else:
        # Legacy
        texts = [t, 'concept: ' + t, 'topic: ' + t]
        vecs = model.encode(texts, normalize_embeddings=True)
        v = (vecs[0] + vecs[1] + vecs[2]) / 3.0

    v = np.asarray(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    _guess_vec_cache.set(cache_key, v)
    return v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0.99 and norm_a < 1.01 and norm_b > 0.99 and norm_b < 1.01:
        return float(np.dot(a, b))
    return float(np.dot(a, b) / (norm_a * norm_b + 1e-12))


# -------------------------
# Scoring
# -------------------------

def score_from_similarity(sim: float) -> int:
    """Map similarité [0,1] vers score [0,1000]"""
    try:
        s = float(sim)
    except Exception:
        s = 0.0

    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0

    # Mapping par paliers calibré pour E5, échelle 0-1000
    if s <= 0.80:
        sc = int(round((s / 0.80) * 40.0)) if s > 0 else 0
        return max(0, min(40, sc))

    if s <= 0.86:
        t = (s - 0.80) / (0.86 - 0.80 + SIM_EPS)
        return int(round(40 + t * (240 - 40)))

    if s <= 0.91:
        t = (s - 0.86) / (0.91 - 0.86 + SIM_EPS)
        return int(round(240 + t * (540 - 240)))

    if s <= 0.95:
        t = (s - 0.91) / (0.95 - 0.91 + SIM_EPS)
        return int(round(540 + t * (790 - 540)))

    if s <= 0.98:
        t = (s - 0.95) / (0.98 - 0.95 + SIM_EPS)
        return int(round(790 + t * (940 - 790)))

    t = (s - 0.98) / (1.0 - 0.98 + SIM_EPS)
    return int(round(940 + t * (1000 - 940)))


def best_anchor_similarity(prepared: PreparedPuzzle, guess_vec: np.ndarray) -> tuple:
    """Trouve la meilleure similarité parmi les ancres"""
    if not prepared.anchor_vecs:
        return 0.0, prepared.canonical_norm

    # Canonical
    try:
        s_c = float(np.dot(guess_vec, prepared.anchor_vecs[0])) * float(
            prepared.anchors_weight[0] if prepared.anchors_weight else 1.0
        )
    except Exception:
        s_c = 0.0

    best_sim = s_c
    best_norm = prepared.canonical_norm

    # Autres ancres
    if len(prepared.anchor_vecs) > 1:
        for idx in range(1, len(prepared.anchor_vecs)):
            a_vec = prepared.anchor_vecs[idx]
            a_norm = prepared.anchors_norm[idx] if idx < len(prepared.anchors_norm) else prepared.canonical_norm
            w = prepared.anchors_weight[idx] if idx < len(prepared.anchors_weight) else 1.0
            try:
                sim = float(np.dot(guess_vec, a_vec)) * float(w)
            except Exception:
                sim = 0.0
            if sim > best_sim:
                best_sim = sim
                best_norm = a_norm

    return best_sim, best_norm


def is_exact_solution(prepared: PreparedPuzzle, normalized_guess: str) -> bool:
    if normalized_guess == prepared.canonical_norm:
        return True
    if normalized_guess in prepared.accepted_norm_set:
        return True
    return False


def score_guess(prepared: PreparedPuzzle, guess_raw: str) -> dict:
    """Évalue une proposition"""
    g_key = normalize_text(guess_raw)

    if not g_key:
        return {'ok': False, 'error': 'empty'}

    # Match exact = victoire
    if is_exact_solution(prepared, g_key):
        return {
            'ok': True,
            'normalized': g_key,
            'is_win': True,
            'semantic_score': 100,
        }

    # Dictionary gate
    if USE_DICTIONARY_GATE:
        if not guess_exists_in_dictionary(guess_raw):
            return {
                'ok': False,
                'error': 'not_in_dictionary',
                'normalized': g_key,
                'note': 'not_in_dictionary',
            }

    # Scoring sémantique
    guess_vec = embed_text(g_key, role=EMBED_ROLE_GUESS)
    best_sim, _best_anchor = best_anchor_similarity(prepared, guess_vec)

    semantic_score = score_from_similarity(best_sim)

    # Exact match sur ancre secondaire -> floor de 75
    try:
        if g_key and len(prepared.anchors_norm) > 1:
            idxs = [i for i in range(1, len(prepared.anchors_norm)) if prepared.anchors_norm[i] == g_key]
            if idxs:
                semantic_score = max(int(semantic_score), EXACT_ANCHOR_MIN_SCORE)
    except Exception:
        pass

    return {
        'ok': True,
        'normalized': g_key,
        'is_win': False,
        'semantic_score': int(semantic_score),
    }


# Test
if __name__ == "__main__":
    logger = logging.getLogger("discoverix.engine")
    logger.info("Loading campaigns...")
    campaigns = get_available_campaigns()
    logger.info("Found %d campaigns: %s", len(campaigns), list(campaigns.keys()))

    if campaigns:
        campaign_name = list(campaigns.keys())[0]
        logger.info("Loading puzzles from campaign: %s", campaign_name)
        puzzles = load_puzzles(campaign_name)
        logger.info("Found %d puzzles", len(puzzles))

        if puzzles:
            first_id = puzzles[0].puzzle_id
            logger.info("Testing with puzzle %s", first_id)

            prepared = prepare_puzzle(first_id, campaign_name)
            if prepared:
                logger.info("Canonical answer: %s", prepared.puzzle.work_canonical)
                logger.info("Hints: %s", prepared.puzzle.reveal_steps[:3])
                logger.info("Anchors count: %d", len(prepared.anchors_text))

                test_guess = prepared.puzzle.work_canonical
                result = score_guess(prepared, test_guess)
                logger.info("Test guess: '%s'", test_guess)
                logger.info("Result: score=%s, win=%s", result.get('score'), result.get('is_win'))
