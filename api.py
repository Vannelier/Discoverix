import time
import uuid
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import (
    get_prepared,
    revealed_hints_list,
    next_hint_in,
    current_reveal_step,
    normalize_key,
    score_guess,
    get_model,
)

app = FastAPI(title='Discoverix API', version='0.6')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

TIMEZONE = 'Europe/Brussels'

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore


def _now_local_dt() -> datetime:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(TIMEZONE))
        except Exception:
            pass
    return datetime.now()


def _today_key() -> str:
    return _now_local_dt().strftime('%Y-%m-%d')


SESSIONS_TTL_SECONDS = 60 * 60 * 30
MAX_SESSIONS = 5000
MIN_GUESS_INTERVAL_SECONDS = 0.35

_sessions: Dict[str, dict] = {}


def _now() -> float:
    return time.time()


def _cleanup_sessions() -> None:
    if not _sessions:
        return

    t = _now()
    today = _today_key()
    to_del = []

    for sid, s in list(_sessions.items()):
        updated_at = float(s.get('updated_at', s.get('created_at', t)))
        if (t - updated_at) > SESSIONS_TTL_SECONDS:
            to_del.append(sid)
            continue

        puzzle_day = (s.get('puzzle_day') or '').strip()
        if puzzle_day and puzzle_day != today:
            to_del.append(sid)

    for sid in to_del:
        _sessions.pop(sid, None)


def _touch_session(s: dict) -> None:
    s['updated_at'] = _now()


def _unwrap_hints(prepared, guess_count_unique: int) -> tuple[list[dict], int, int]:
    res = revealed_hints_list(prepared.reveal_steps, guess_count_unique)
    hints = list(res) if res else []
    step_target = int(current_reveal_step(guess_count_unique))
    step_last_shown = int(hints[-1]['step']) if hints else -1
    return hints, step_target, step_last_shown


def _win_reveal_payload(prepared, hints_after: list[dict]) -> dict:
    return {
        'work_canonical': prepared.work_canonical,
        'character_name_full': prepared.character_name_full,
        'wikidata_qid': prepared.wikidata_qid,
        'revealed_hints': hints_after,
    }


class StartRequest(BaseModel):
    puzzle_id: str


class StartResponse(BaseModel):
    session_id: str
    puzzle_id: str
    puzzle_day: str
    guess_count_unique: int
    is_win: bool
    reveal_step_target: int
    reveal_step_last_shown: int
    revealed_hints: list[dict]
    next_hint_in: int | None


class GuessRequest(BaseModel):
    session_id: str
    guess_text: str


class GuessResponse(BaseModel):
    ok: bool
    error: str | None
    session_id: str
    puzzle_id: str
    puzzle_day: str
    guess_count_unique: int
    normalized: str | None
    score: int | None
    is_win: bool
    is_duplicate: bool
    reveal_step_target: int
    reveal_step_last_shown: int
    revealed_hints: list[dict]
    next_hint_in: int | None
    win_reveal: dict | None


class SessionResponse(BaseModel):
    session_id: str
    puzzle_id: str
    puzzle_day: str
    guess_count_unique: int
    is_win: bool
    reveal_step_target: int
    reveal_step_last_shown: int
    revealed_hints: list[dict]
    next_hint_in: int | None
    guesses: list[dict]
    win_reveal: dict | None


@app.on_event('startup')
def _warmup():
    # Charge le modele au demarrage pour eviter la latence du premier appel.
    try:
        get_model()
    except Exception:
        pass


@app.get('/api/v1/health')
def health():
    _cleanup_sessions()
    return {'ok': True, 'sessions': len(_sessions), 'puzzle_day': _today_key()}


@app.post('/api/v1/start', response_model=StartResponse)
def start(req: StartRequest):
    _cleanup_sessions()
    if len(_sessions) >= MAX_SESSIONS:
        raise HTTPException(503, 'trop de sessions actives')

    puzzle_id = (req.puzzle_id or '').strip()
    if not puzzle_id:
        raise HTTPException(400, 'puzzle_id manquant')

    try:
        prepared = get_prepared(puzzle_id)
    except KeyError:
        raise HTTPException(404, 'puzzle introuvable')
    except Exception as e:
        raise HTTPException(500, f'erreur preparation puzzle: {e}')

    session_id = str(uuid.uuid4())
    created_at = _now()
    puzzle_day = _today_key()

    _sessions[session_id] = {
        'puzzle_id': puzzle_id,
        'puzzle_day': puzzle_day,
        'guess_count_unique': 0,
        'guesses': [],
        'guesses_by_norm': {},
        'is_win': False,
        'created_at': created_at,
        'updated_at': created_at,
        'last_guess_at': 0.0,
    }

    hints, step_target, step_last_shown = _unwrap_hints(prepared, 0)

    return StartResponse(
        session_id=session_id,
        puzzle_id=puzzle_id,
        puzzle_day=puzzle_day,
        guess_count_unique=0,
        is_win=False,
        reveal_step_target=step_target,
        reveal_step_last_shown=step_last_shown,
        revealed_hints=hints,
        next_hint_in=next_hint_in(0),
    )


@app.post('/api/v1/guess', response_model=GuessResponse)
def guess(req: GuessRequest):
    _cleanup_sessions()

    s = _sessions.get(req.session_id)
    if not s:
        raise HTTPException(404, 'session introuvable')

    _touch_session(s)

    if s['is_win']:
        raise HTTPException(409, 'partie deja terminee')

    if (s.get('puzzle_day') or '').strip() != _today_key():
        raise HTTPException(409, 'session expiree (nouveau puzzle)')

    t = _now()
    if (t - float(s.get('last_guess_at', 0.0))) < MIN_GUESS_INTERVAL_SECONDS:
        raise HTTPException(429, 'trop rapide')
    s['last_guess_at'] = t

    puzzle_id = s['puzzle_id']
    prepared = get_prepared(puzzle_id)

    norm = normalize_key(req.guess_text)
    count_before = int(s['guess_count_unique'])

    if not norm:
        hints, step_target, step_last_shown = _unwrap_hints(prepared, count_before)
        return GuessResponse(
            ok=False,
            error='empty',
            session_id=req.session_id,
            puzzle_id=puzzle_id,
            puzzle_day=s['puzzle_day'],
            guess_count_unique=count_before,
            normalized=None,
            score=None,
            is_win=bool(s['is_win']),
            is_duplicate=False,
            reveal_step_target=step_target,
            reveal_step_last_shown=step_last_shown,
            revealed_hints=hints,
            next_hint_in=next_hint_in(count_before),
            win_reveal=None,
        )

    prev = s['guesses_by_norm'].get(norm)
    if prev is not None:
        hints, step_target, step_last_shown = _unwrap_hints(prepared, count_before)
        return GuessResponse(
            ok=True,
            error=None,
            session_id=req.session_id,
            puzzle_id=puzzle_id,
            puzzle_day=s['puzzle_day'],
            guess_count_unique=count_before,
            normalized=prev['normalized'],
            score=int(prev['score']),
            is_win=bool(s['is_win']),
            is_duplicate=True,
            reveal_step_target=step_target,
            reveal_step_last_shown=step_last_shown,
            revealed_hints=hints,
            next_hint_in=next_hint_in(count_before),
            win_reveal=None,
        )

    r = score_guess(prepared, req.guess_text)

    if not r.get('ok'):
        hints, step_target, step_last_shown = _unwrap_hints(prepared, count_before)
        return GuessResponse(
            ok=False,
            error=str(r.get('error', 'erreur')),
            session_id=req.session_id,
            puzzle_id=puzzle_id,
            puzzle_day=s['puzzle_day'],
            guess_count_unique=count_before,
            normalized=r.get('normalized') or norm,
            score=None,
            is_win=False,
            is_duplicate=False,
            reveal_step_target=step_target,
            reveal_step_last_shown=step_last_shown,
            revealed_hints=hints,
            next_hint_in=next_hint_in(count_before),
            win_reveal=None,
        )

    # Consomme un essai unique uniquement si ok
    s['guess_count_unique'] = count_before + 1
    entry = {'raw': req.guess_text, 'normalized': r['normalized'], 'score': int(r['score'])}
    s['guesses'].append(entry)
    s['guesses_by_norm'][norm] = entry

    if r.get('is_win'):
        s['is_win'] = True

    count_after = int(s['guess_count_unique'])
    hints_after, step_target_after, step_last_shown_after = _unwrap_hints(prepared, count_after)
    win_reveal = _win_reveal_payload(prepared, hints_after) if s['is_win'] else None

    return GuessResponse(
        ok=True,
        error=None,
        session_id=req.session_id,
        puzzle_id=puzzle_id,
        puzzle_day=s['puzzle_day'],
        guess_count_unique=count_after,
        normalized=r['normalized'],
        score=int(r['score']),
        is_win=bool(s['is_win']),
        is_duplicate=False,
        reveal_step_target=step_target_after,
        reveal_step_last_shown=step_last_shown_after,
        revealed_hints=hints_after,
        next_hint_in=next_hint_in(count_after),
        win_reveal=win_reveal,
    )


@app.get('/api/v1/session/{session_id}', response_model=SessionResponse)
def session(session_id: str):
    _cleanup_sessions()

    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(404, 'session introuvable')

    _touch_session(s)

    prepared = get_prepared(s['puzzle_id'])
    count = int(s['guess_count_unique'])
    hints, step_target, step_last_shown = _unwrap_hints(prepared, count)
    win_reveal = _win_reveal_payload(prepared, hints) if bool(s['is_win']) else None

    return SessionResponse(
        session_id=session_id,
        puzzle_id=s['puzzle_id'],
        puzzle_day=s.get('puzzle_day') or _today_key(),
        guess_count_unique=count,
        is_win=bool(s['is_win']),
        reveal_step_target=step_target,
        reveal_step_last_shown=step_last_shown,
        revealed_hints=hints,
        next_hint_in=next_hint_in(count),
        guesses=s['guesses'],
        win_reveal=win_reveal,
    )
