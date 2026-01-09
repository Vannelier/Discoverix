"""Discoverix API"""
import time
import uuid
import os
import json
import threading
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from .engine import prepare_puzzle, load_puzzles, get_available_campaigns, normalize_text, score_guess, get_model, get_puzzle_ids_for_campaign

TIMEZONE = 'Europe/Brussels'
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# Logger
logger = logging.getLogger("discoverix")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

def _now_local_dt() -> datetime:
    try:
        return datetime.now(ZoneInfo(TIMEZONE)) if ZoneInfo else datetime.now()
    except Exception:
        return datetime.now()

def _today_key() -> str:
    return _now_local_dt().strftime('%Y-%m-%d')

SESSIONS_FILE = os.getenv('SESSIONS_FILE', 'sessions.json')
SESSIONS_TTL_SECONDS = 60 * 60 * 30
MAX_SESSIONS = 5000
MIN_GUESS_INTERVAL_SECONDS = 0.35
SESSIONS_SAVE_DEBOUNCE_SECONDS = float(os.getenv('SESSIONS_SAVE_DEBOUNCE_SECONDS', '0') or 0)
SESSIONS_WRITE_FSYNC = (os.getenv('SESSIONS_WRITE_FSYNC', '1') or '1').strip() not in ('0', 'false', 'False')

# Constantes de la run de survie
INITIAL_POINTS = 100
GUESS_COST = 1
HINT_BASE_COST = 10  # Premier indice = 10, puis +5 à chaque fois (10, 15, 20, 25...)
HINT_COST_INCREMENT = 5
# Bonus indice trouvé = coût de l'indice (progressif: 10, 15, 20, 25...)
PERFECT_BONUS = 30  # +30 points si résolu en 1 coup
JACKPOT_BONUS = 20  # +20 points si aucun indice acheté
NORMAL_WIN_BONUS = 10  # +10 points pour victoire normale
RESURRECTION_POINTS = 10  # Points après victoire à 0
PUZZLES_PER_CAMPAIGN = 10  # Nombre de puzzles pour compléter une campagne
XP_PER_PUZZLE = 1  # XP de base par puzzle résolu
XP_JACKPOT_BONUS = 1  # XP bonus pour jackpot
XP_FIRST_GUESS_BONUS = 1  # XP bonus pour premier coup

_sessions_lock = threading.RLock()
_last_sessions_save_at: float = 0.0
_sessions: Dict[str, dict] = {}
_runs: Dict[str, dict] = {}  # Stockage des runs en cours
_players: Dict[str, dict] = {}  # Stockage des profils joueurs (XP, campagnes débloquées)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
        campaigns = get_available_campaigns()
        logger.info("Found %d campaigns: %s", len(campaigns), list(campaigns.keys()))
        if campaigns:
            load_puzzles(list(campaigns.keys())[0])
    except Exception as e:
        logger.exception("Warmup error")
    try:
        _load_sessions()
    except Exception as e:
        logger.exception("Session loading error")
    yield
    try:
        _save_sessions(force=True)
    except Exception as e:
        logger.exception("Shutdown save error")


app = FastAPI(title='Discoverix API', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

def _load_sessions() -> None:
    global _sessions
    try:
        if not os.path.isfile(SESSIONS_FILE):
            with _sessions_lock:
                _sessions = {}
            return

        with open(SESSIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            data = {}

        cleaned: Dict[str, dict] = {}
        for sid, s in data.items():
            if isinstance(sid, str) and isinstance(s, dict):
                cleaned[sid] = s

        with _sessions_lock:
            _sessions = cleaned

        logger.info("Loaded %d sessions from disk", len(_sessions))

    except Exception as e:
        logger.exception("Error loading sessions")
        with _sessions_lock:
            _sessions = {}


def _save_sessions(force: bool = False) -> None:
    """Persist sessions vers fichier (atomique)"""
    global _last_sessions_save_at

    try:
        now = _now()
        with _sessions_lock:
            if (not force) and SESSIONS_SAVE_DEBOUNCE_SECONDS > 0:
                if (now - _last_sessions_save_at) < SESSIONS_SAVE_DEBOUNCE_SECONDS:
                    return

            payload = json.dumps(_sessions, ensure_ascii=False, separators=(',', ':'))
            _last_sessions_save_at = now

        # Écriture atomique
        abs_path = os.path.abspath(SESSIONS_FILE)
        dir_path = os.path.dirname(abs_path) or '.'
        os.makedirs(dir_path, exist_ok=True)

        tmp_path = f"{abs_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"

        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(payload)
                f.flush()
                if SESSIONS_WRITE_FSYNC:
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass

            os.replace(tmp_path, abs_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    except Exception as e:
        logger.exception("Error saving sessions")


def _now() -> float:
    return time.time()


def _cleanup_sessions() -> None:
    """Nettoie les sessions expirées"""
    today = _today_key()
    t = _now()
    to_del: list = []

    with _sessions_lock:
        for sid, s in list(_sessions.items()):
            updated_at = float(s.get('updated_at', 0.0))
            if (t - updated_at) > SESSIONS_TTL_SECONDS:
                to_del.append(sid)
                continue

            puzzle_day = (s.get('puzzle_day') or '').strip()
            if puzzle_day and puzzle_day != today:
                to_del.append(sid)

        for sid in to_del:
            _sessions.pop(sid, None)

    if to_del:
        logger.info("Cleaned up %d expired sessions", len(to_del))
        _save_sessions()


def _touch_session(s: dict) -> None:
    """Met à jour le timestamp de session"""
    s['updated_at'] = _now()


# -------------------------
# Hints logic (NEW)
# -------------------------

def _calculate_hint_cost(index: int) -> int:
    """Calcule le coût d'un indice selon sa position (10, 15, 20, 25...)."""
    return HINT_BASE_COST + (index * HINT_COST_INCREMENT)

def _get_hints_for_api(prepared, session: dict) -> list:
    """Prépare la liste des indices (révélés ou masqués) pour l'API."""
    all_hints = list(prepared.puzzle.reveal_steps or [])
    revealed_indices = set(session.get('revealed_hints_indices', []))

    # Map column names to simple labels (keep original if already in French)
    label_map = {
        'Category': 'Catégorie',
        'instance_of': 'Type',
        'country_of_origin': 'Pays',
        'description_en': 'Description',
        'ANNÉE': 'ANNÉE',
        'AUTEUR': 'AUTEUR',
        'RÉALISATEUR': 'RÉALISATEUR',
        'CRÉATEUR': 'CRÉATEUR',
        'COMPOSITEUR': 'COMPOSITEUR',
        'INTERPRÈTE': 'INTERPRÈTE',
        'PAYS': 'PAYS',
        'GENRE': 'GENRE',
        'COLLECTION': 'COLLECTION',
        'SÉRIE': 'SÉRIE',
        'ACTEUR': 'ACTEUR',
        'SCÉNARISTE': 'SCÉNARISTE',
        'PRODUCTEUR': 'PRODUCTEUR',
        'STUDIO': 'STUDIO',
        'ÉDITEUR': 'ÉDITEUR',
        'PLATEFORME': 'PLATEFORME',
        'PRIX': 'PRIX',
        'LANGUE': 'LANGUE',
        'DURÉE': 'DURÉE',
        'ÉPISODES': 'ÉPISODES',
        'SAISONS': 'SAISONS',
        'PAGES': 'PAGES',
        'LIEU': 'LIEU',
        'MÉTIER': 'MÉTIER',
        'DOMAINE': 'DOMAINE',
        'CAPITALE': 'CAPITALE',
        'POPULATION': 'POPULATION',
    }

    api_hints = []
    for i, hint in enumerate(all_hints):
        is_revealed = i in revealed_indices
        hint_type = hint.get('type', 'Indice')
        simple_label = label_map.get(hint_type, hint_type)

        # Coût progressif
        cost = _calculate_hint_cost(i)

        api_hints.append({
            'hint_id': str(i),
            'index': i,
            'type': simple_label,
            'text': hint.get('text', '') if is_revealed else '?',
            'revealed': is_revealed,
            'cost': cost
        })
    return api_hints


def _all_hints_payload(prepared) -> list:
    """Tous les indices pour l'écran de victoire."""
    steps = list(prepared.puzzle.reveal_steps or [])
    out: list = []
    for i, hint in enumerate(steps):
        text = hint.get('text', '').strip()
        if text:
            out.append({
                'type': hint.get('type', 'Indice'),
                'text': text
            })
    return out


def _win_reveal_payload(prepared) -> dict:
    """Payload de victoire"""
    p = prepared.puzzle
    return {
        'work_canonical': p.work_canonical,
        'character_name_full': p.character_name_full,
        'revealed_hints': _all_hints_payload(prepared),
        'wikidata_qid': p.wikidata_qid,
        'wikidata_url': p.wikidata_url or '',
    }


async def _get_prepared_from_request(puzzle_id: str, campaign: str = None):
    """Prépare un puzzle depuis l'ID et la campagne (exécuté en thread pour éviter de bloquer l'event loop)."""
    pid_raw = (puzzle_id or '').strip()
    if not pid_raw:
        raise HTTPException(status_code=400, detail='puzzle_id manquant')
    try:
        pid = int(pid_raw)
    except Exception:
        raise HTTPException(status_code=400, detail='puzzle_id invalide')

    try:
        return await asyncio.to_thread(prepare_puzzle, pid, campaign)
    except KeyError:
        raise HTTPException(status_code=404, detail=f'puzzle {pid} introuvable dans campagne {campaign}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'erreur preparation puzzle: {e}')


# -------------------------
# Pydantic models
# -------------------------

class StartRunRequest(BaseModel):
    campaign: str
    is_free: bool = False

class StartRunResponse(BaseModel):
    run_id: str
    campaign: str
    points: int
    puzzles_solved: int
    puzzle_ids: list

class StartRequest(BaseModel):
    run_id: str = None
    puzzle_id: str = None

class StartResponse(BaseModel):
    session_id: str
    puzzle_id: str
    campaign: str
    puzzle_day: str
    player_score: int
    guess_count_unique: int
    is_win: bool
    revealed_hints: list
    next_hint_in: int | None

class GuessRequest(BaseModel):
    session_id: str
    guess_text: str

class GuessResponse(BaseModel):
    ok: bool
    error: str | None
    note: str | None
    session_id: str
    puzzle_id: str
    puzzle_day: str
    player_score: int
    gained_points: int | None
    guess_count_unique: int
    normalized: str | None
    score: int | None
    is_win: bool
    is_duplicate: bool
    revealed_hints: list
    next_hint_in: int | None
    win_reveal: dict | None
    jackpot_bonus: bool = False
    perfect_bonus: bool = False
    normal_win_bonus: bool = False
    resurrection_bonus: bool = False
    bonus_points: int = 0

class SessionResponse(BaseModel):
    session_id: str
    puzzle_id: str
    campaign: str
    puzzle_day: str
    player_score: int
    guess_count_unique: int
    is_win: bool
    revealed_hints: list
    next_hint_in: int | None
    guesses: list
    win_reveal: dict | None

class RevealHintRequest(BaseModel):
    session_id: str
    hint_index: int

class FreeCampaignResponse(BaseModel):
    available: bool
    last_used_date: str | None



# -------------------------
# Run & Player Management
# -------------------------

def _get_or_create_player(player_id: str = 'default') -> dict:
    """Récupère ou crée un profil joueur."""
    with _sessions_lock:
        if player_id not in _players:
            _players[player_id] = {
                'player_id': player_id,
                'xp': 0,
                'level': 1,
                'last_free_campaign_date': None,
                'unlocked_campaigns': [],
                'created_at': _now(),
            }
        return _players[player_id]

def _can_use_free_campaign(player: dict) -> bool:
    """Vérifie si le joueur peut utiliser une campagne gratuite aujourd'hui."""
    today = _today_key()
    last_date = player.get('last_free_campaign_date')
    return last_date != today

def _mark_free_campaign_used(player: dict) -> None:
    """Marque la campagne gratuite comme utilisée aujourd'hui."""
    player['last_free_campaign_date'] = _today_key()

def _get_run(run_id: str) -> dict:
    """Récupère une run par ID."""
    with _sessions_lock:
        return _runs.get(run_id)

def _create_run(campaign: str, player_id: str = 'default') -> dict:
    """Crée une nouvelle run."""
    run_id = str(uuid.uuid4())
    puzzle_ids = get_puzzle_ids_for_campaign(campaign)

    run_data = {
        'run_id': run_id,
        'player_id': player_id,
        'campaign': campaign,
        'points': INITIAL_POINTS,
        'puzzles_solved': 0,
        'puzzle_ids': puzzle_ids,
        'current_puzzle_index': 0,
        'hints_bought_count': 0,
        'hints_discovered_count': 0,
        'jackpots_count': 0,
        'total_guesses': 0,
        'xp_earned': 0,
        'created_at': _now(),
        'updated_at': _now(),
    }

    with _sessions_lock:
        _runs[run_id] = run_data

    return run_data

def _update_run_on_puzzle_complete(run: dict, puzzle_session: dict) -> dict:
    """Met à jour la run après la complétion d'un puzzle."""
    with _sessions_lock:
        run['puzzles_solved'] += 1
        run['current_puzzle_index'] += 1
        run['updated_at'] = _now()

        # Calcul des bonus et ajustement des points
        final_points = puzzle_session['score']
        hints_bought = len([i for i in puzzle_session.get('hints_bought', [])])
        guess_count = puzzle_session['guess_count_unique']

        jackpot = False
        perfect = False
        resurrection = False
        normal_win = False
        xp_bonus = 0
        bonus_points = 0

        # Victoire dramatique (résurrection)
        if final_points <= 0:
            final_points = RESURRECTION_POINTS
            resurrection = True
        else:
            # Premier coup parfait → +30 points
            if guess_count == 1:
                bonus_points = PERFECT_BONUS
                final_points += bonus_points
                perfect = True
                xp_bonus += XP_FIRST_GUESS_BONUS
            # Jackpot sans indice acheté → +20 points
            elif hints_bought == 0:
                bonus_points = JACKPOT_BONUS
                final_points += bonus_points
                jackpot = True
                xp_bonus += XP_JACKPOT_BONUS
                run['jackpots_count'] += 1
            # Victoire normale → +10 points
            else:
                bonus_points = NORMAL_WIN_BONUS
                final_points += bonus_points
                normal_win = True

        run['points'] = final_points
        puzzle_session['score'] = final_points

        # XP
        xp_gain = XP_PER_PUZZLE + xp_bonus
        run['xp_earned'] += xp_gain

        return {
            'jackpot': jackpot,
            'perfect': perfect,
            'resurrection': resurrection,
            'normal_win': normal_win,
            'bonus_points': bonus_points,
            'xp_gain': xp_gain,
            'final_points': final_points
        }

def _check_run_status(run: dict) -> dict:
    """Vérifie le statut de la run (game over, run complete, continue)."""
    points = run['points']
    puzzles_solved = run['puzzles_solved']
    total_puzzles = len(run['puzzle_ids'])

    if points <= 0:
        return {'status': 'game_over', 'reason': 'no_points'}

    if puzzles_solved >= PUZZLES_PER_CAMPAIGN or run['current_puzzle_index'] >= total_puzzles:
        return {'status': 'run_complete', 'reason': 'campaign_complete'}

    return {'status': 'active', 'reason': 'continue'}

# -------------------------
# Endpoints
# -------------------------

@app.get('/api/v1/health')
async def health():
    """Santé de l'API"""
    _cleanup_sessions()
    campaigns = get_available_campaigns()
    return {
        'ok': True,
        'sessions': len(_sessions),
        'puzzle_day': _today_key(),
        'campaigns': list(campaigns.keys())
    }


@app.get('/api/v1/campaigns')
async def list_campaigns():
    """Liste toutes les campagnes disponibles"""
    campaigns = get_available_campaigns()
    return {
        'campaigns': [
            {'name': name, 'file': os.path.basename(path)}
            for name, path in campaigns.items()
        ]
    }


@app.get('/api/v1/campaigns/{campaign_name}/puzzles')
async def list_puzzles_in_campaign(campaign_name: str):
    """Retourne la liste des IDs de puzzles pour une campagne"""
    try:
        puzzle_ids = get_puzzle_ids_for_campaign(campaign_name)
        return {'campaign': campaign_name, 'puzzle_ids': puzzle_ids}
    except Exception as e:
        raise HTTPException(404, f'Campagne non trouvée ou erreur: {e}')


@app.get('/api/v1/free_campaign_available', response_model=FreeCampaignResponse)
async def check_free_campaign():
    """Vérifie si une campagne gratuite est disponible aujourd'hui"""
    player = _get_or_create_player()
    available = _can_use_free_campaign(player)
    return FreeCampaignResponse(
        available=available,
        last_used_date=player.get('last_free_campaign_date')
    )


@app.post('/api/v1/start_run', response_model=StartRunResponse)
async def start_run(req: StartRunRequest):
    """Démarre une nouvelle run de survie"""
    _cleanup_sessions()

    player = _get_or_create_player()

    # Vérifier si campagne gratuite
    if req.is_free:
        if not _can_use_free_campaign(player):
            raise HTTPException(403, 'Campagne gratuite déjà utilisée aujourd\'hui')
        _mark_free_campaign_used(player)

    # Créer la run
    run = _create_run(req.campaign)

    return StartRunResponse(
        run_id=run['run_id'],
        campaign=run['campaign'],
        points=run['points'],
        puzzles_solved=run['puzzles_solved'],
        puzzle_ids=run['puzzle_ids']
    )


@app.post('/api/v1/start', response_model=StartResponse)
async def start(req: StartRequest):
    """Démarre une nouvelle session de puzzle dans une run"""
    _cleanup_sessions()

    with _sessions_lock:
        if len(_sessions) >= MAX_SESSIONS:
            raise HTTPException(503, 'trop de sessions actives')

    # Récupérer la run
    run_id = req.run_id
    if not run_id:
        raise HTTPException(400, 'run_id requis')

    run = _get_run(run_id)
    if not run:
        raise HTTPException(404, 'run introuvable')

    # Vérifier le statut de la run
    run_status = _check_run_status(run)
    if run_status['status'] != 'active':
        raise HTTPException(409, f'Run terminée: {run_status["reason"]}')

    # Déterminer le puzzle à charger
    campaign = run['campaign']
    if req.puzzle_id:
        # Puzzle spécifique demandé
        puzzle_id = req.puzzle_id
    else:
        # Puzzle suivant dans la run
        if run['current_puzzle_index'] >= len(run['puzzle_ids']):
            raise HTTPException(409, 'Plus de puzzles dans cette campagne')
        puzzle_id = str(run['puzzle_ids'][run['current_puzzle_index']])

    prepared = await _get_prepared_from_request(puzzle_id, campaign)

    session_id = str(uuid.uuid4())
    created_at = _now()
    puzzle_day = _today_key()

    session_data = {
        'session_id': session_id,
        'run_id': run_id,
        'puzzle_id': puzzle_id,
        'campaign': campaign,
        'puzzle_day': puzzle_day,
        'score': run['points'],
        'guess_count_unique': 0,
        'guesses': [],
        'guesses_by_norm': {},
        'is_win': False,
        'revealed_hints_indices': [],
        'hints_bought': [],  # Track des indices achetés pour le jackpot
        'created_at': created_at,
        'updated_at': created_at,
        'last_guess_at': 0.0,
    }

    with _sessions_lock:
        _sessions[session_id] = session_data

    _save_sessions()

    hints_payload = _get_hints_for_api(prepared, session_data)

    return StartResponse(
        session_id=session_id,
        puzzle_id=puzzle_id,
        campaign=campaign,
        puzzle_day=puzzle_day,
        player_score=session_data['score'],
        guess_count_unique=0,
        is_win=False,
        revealed_hints=hints_payload,
        next_hint_in=None
    )


@app.post('/api/v1/guess', response_model=GuessResponse)
async def guess(req: GuessRequest):
    """Traite une proposition"""
    _cleanup_sessions()

    with _sessions_lock:
        s = _sessions.get(req.session_id)
        if not s:
            raise HTTPException(404, 'session introuvable')
        _touch_session(s)

    if s['is_win']:
        raise HTTPException(409, 'partie deja terminee')

    if (s.get('puzzle_day') or '').strip() != _today_key():
        raise HTTPException(409, 'session expiree (nouveau puzzle)')

    t = _now()
    with _sessions_lock:
        if (t - float(s.get('last_guess_at', 0.0))) < MIN_GUESS_INTERVAL_SECONDS:
            raise HTTPException(429, 'trop rapide')
        s['last_guess_at'] = t

    puzzle_id = s['puzzle_id']
    campaign = s.get('campaign')
    run_id = s.get('run_id')
    run = _get_run(run_id) if run_id else None

    prepared = await _get_prepared_from_request(puzzle_id, campaign)

    norm = normalize_text(req.guess_text)
    count_before = int(s['guess_count_unique'])

    # Chaque essai coûte 1 point
    with _sessions_lock:
        s['score'] = max(0, s['score'] - GUESS_COST)
        if run:
            run['total_guesses'] += 1

    if not norm:
        hints = _get_hints_for_api(prepared, s)
        return GuessResponse(
            ok=False, error='empty', note=None, session_id=req.session_id,
            puzzle_id=puzzle_id, puzzle_day=s['puzzle_day'],
            player_score=s['score'], gained_points=0,
            guess_count_unique=count_before, normalized=None, score=None,
            is_win=bool(s['is_win']), is_duplicate=False,
            revealed_hints=hints, next_hint_in=None, win_reveal=None
        )

    # Check doublon
    prev = s['guesses_by_norm'].get(norm)
    if prev is not None:
        hints = _get_hints_for_api(prepared, s)
        return GuessResponse(
            ok=True, error=None, note=None, session_id=req.session_id,
            puzzle_id=puzzle_id, puzzle_day=s['puzzle_day'],
            player_score=s['score'], gained_points=0,
            guess_count_unique=count_before, normalized=prev['normalized'],
            score=int(prev['semantic_score']), is_win=bool(s['is_win']),
            is_duplicate=True, revealed_hints=hints, next_hint_in=None, win_reveal=None
        )

    # Score guess
    r = await asyncio.to_thread(score_guess, prepared, req.guess_text)

    if not r.get('ok'):
        hints = _get_hints_for_api(prepared, s)
        return GuessResponse(
            ok=False, error=str(r.get('error', 'erreur')),
            note=str(r.get('note')) if r.get('note') is not None else None,
            session_id=req.session_id, puzzle_id=puzzle_id, puzzle_day=s['puzzle_day'],
            player_score=s['score'], gained_points=0, guess_count_unique=count_before,
            normalized=r.get('normalized') or norm, score=None,
            is_win=False, is_duplicate=False, revealed_hints=hints, next_hint_in=None, win_reveal=None
        )

    # Valide : consomme un essai
    with _sessions_lock:
        s['guess_count_unique'] = count_before + 1
        entry = {
            'raw': req.guess_text,
            'normalized': r['normalized'],
            'semantic_score': int(r['semantic_score']),
            'attempt': int(s['guess_count_unique']),
        }
        s['guesses'].append(entry)
        s['guesses_by_norm'][norm] = entry

        if r.get('is_win'):
            s['is_win'] = True

    # Vérifier si le guess révèle un indice naturellement
    gained_points = 0
    all_hints = list(prepared.puzzle.reveal_steps or [])
    revealed_indices = s.setdefault('revealed_hints_indices', [])

    for i, hint in enumerate(all_hints):
        if i not in revealed_indices:
            hint_text_norm = normalize_text(hint.get('text', ''))
            if hint_text_norm and norm == hint_text_norm:
                revealed_indices.append(i)
                # Gagner le coût de l'indice au lieu d'un bonus fixe
                hint_cost = _calculate_hint_cost(i)
                gained_points += hint_cost
                if run:
                    run['hints_discovered_count'] += 1

    if gained_points > 0:
        with _sessions_lock:
            s['score'] += gained_points

    # Bonus de victoire
    jackpot_bonus = False
    perfect_bonus = False
    normal_win_bonus = False
    resurrection_bonus = False
    bonus_points = 0

    if s['is_win'] and run:
        bonus_result = _update_run_on_puzzle_complete(run, s)
        jackpot_bonus = bonus_result['jackpot']
        perfect_bonus = bonus_result['perfect']
        normal_win_bonus = bonus_result['normal_win']
        resurrection_bonus = bonus_result['resurrection']
        bonus_points = bonus_result['bonus_points']

        # Update player XP
        player = _get_or_create_player()
        player['xp'] += bonus_result['xp_gain']

    with _sessions_lock:
        _save_sessions()

    hints_after = _get_hints_for_api(prepared, s)
    win_reveal = _win_reveal_payload(prepared) if s['is_win'] else None

    return GuessResponse(
        ok=True, error=None, note=str(r.get('note')) if r.get('note') is not None else None,
        session_id=req.session_id, puzzle_id=puzzle_id, puzzle_day=s['puzzle_day'],
        player_score=s['score'], gained_points=gained_points,
        guess_count_unique=s['guess_count_unique'], normalized=r['normalized'],
        score=int(r['semantic_score']), is_win=bool(s['is_win']), is_duplicate=False,
        revealed_hints=hints_after, next_hint_in=None, win_reveal=win_reveal,
        jackpot_bonus=jackpot_bonus,
        perfect_bonus=perfect_bonus,
        normal_win_bonus=normal_win_bonus,
        resurrection_bonus=resurrection_bonus,
        bonus_points=bonus_points
    )


@app.post('/api/v1/reveal_hint', response_model=SessionResponse)
async def reveal_hint(req: RevealHintRequest):
    """Révèle un indice en échange de points"""
    with _sessions_lock:
        s = _sessions.get(req.session_id)
        if not s:
            raise HTTPException(404, 'session introuvable')
        _touch_session(s)

    if s['is_win']:
        raise HTTPException(409, 'partie deja terminee')

    run_id = s.get('run_id')
    run = _get_run(run_id) if run_id else None

    puzzle_id = s['puzzle_id']
    campaign = s.get('campaign')
    prepared = await _get_prepared_from_request(puzzle_id, campaign)

    all_hints = list(prepared.puzzle.reveal_steps or [])
    hint_index = req.hint_index

    if not (0 <= hint_index < len(all_hints)):
        raise HTTPException(400, 'indice invalide')

    revealed_indices = s.setdefault('revealed_hints_indices', [])
    if hint_index in revealed_indices:
        # Déjà révélé
        hints = _get_hints_for_api(prepared, s)
        win_reveal = _win_reveal_payload(prepared) if s['is_win'] else None
        return SessionResponse(
            session_id=req.session_id, puzzle_id=puzzle_id, campaign=campaign,
            puzzle_day=s['puzzle_day'], player_score=s['score'],
            guess_count_unique=s['guess_count_unique'], is_win=s['is_win'],
            revealed_hints=hints, next_hint_in=None, guesses=s['guesses'], win_reveal=win_reveal
        )

    # Coût progressif selon l'index
    hint_cost = _calculate_hint_cost(hint_index)
    with _sessions_lock:
        s['score'] = max(0, s['score'] - hint_cost)
        s.setdefault('hints_bought', []).append(hint_index)
        if run:
            run['hints_bought_count'] += 1

    with _sessions_lock:
        revealed_indices.append(hint_index)
        _save_sessions()

    hints = _get_hints_for_api(prepared, s)
    win_reveal = _win_reveal_payload(prepared) if s['is_win'] else None

    return SessionResponse(
        session_id=req.session_id,
        puzzle_id=puzzle_id,
        campaign=campaign or 'Default',
        puzzle_day=s.get('puzzle_day') or _today_key(),
        player_score=s['score'],
        guess_count_unique=s['guess_count_unique'],
        is_win=bool(s['is_win']),
        revealed_hints=hints,
        next_hint_in=None,
        guesses=s['guesses'],
        win_reveal=win_reveal
    )


@app.get('/api/v1/session/{session_id}', response_model=SessionResponse)
async def session(session_id: str):
    """Retourne l'état d'une session"""
    _cleanup_sessions()

    with _sessions_lock:
        s = _sessions.get(session_id)
        if not s:
            raise HTTPException(404, 'session introuvable')
        _touch_session(s)

    run_id = s.get('run_id')
    run = _get_run(run_id) if run_id else None

    puzzle_id = s['puzzle_id']
    campaign = s.get('campaign')
    prepared = await _get_prepared_from_request(puzzle_id, campaign)

    hints = _get_hints_for_api(prepared, s)
    win_reveal = _win_reveal_payload(prepared) if s['is_win'] else None

    return SessionResponse(
        session_id=session_id,
        puzzle_id=puzzle_id,
        campaign=campaign or 'Default',
        puzzle_day=s.get('puzzle_day') or _today_key(),
        player_score=s.get('score', INITIAL_POINTS),
        guess_count_unique=s['guess_count_unique'],
        is_win=bool(s['is_win']),
        revealed_hints=hints,
        next_hint_in=None,
        guesses=s['guesses'],
        win_reveal=win_reveal
    )


@app.get('/api/v1/run/{run_id}/stats')
async def get_run_stats(run_id: str):
    """Retourne les statistiques d'une run"""
    run = _get_run(run_id)
    if not run:
        raise HTTPException(404, 'run introuvable')

    total_guesses = run.get('total_guesses', 0)
    puzzles_solved = run.get('puzzles_solved', 0)
    avg_guesses = round(total_guesses / puzzles_solved, 1) if puzzles_solved > 0 else 0

    return {
        'run_id': run_id,
        'campaign': run.get('campaign', ''),
        'score': run.get('points', 0),
        'puzzles_solved': puzzles_solved,
        'total_puzzles': len(run.get('puzzle_ids', [])),
        'total_guesses': total_guesses,
        'average_guesses': avg_guesses,
        'hints_bought': run.get('hints_bought_count', 0),
        'hints_discovered': run.get('hints_discovered_count', 0),
        'jackpots': run.get('jackpots_count', 0),
        'xp_earned': run.get('xp_earned', 0)
    }


class SkipPuzzleRequest(BaseModel):
    session_id: str

@app.post('/api/v1/skip_puzzle')
async def skip_puzzle(req: SkipPuzzleRequest):
    """Passe le puzzle actuel en divisant les points par 2 et avance au suivant"""
    with _sessions_lock:
        s = _sessions.get(req.session_id)
        if not s:
            raise HTTPException(404, 'session introuvable')
        _touch_session(s)

    if s['is_win']:
        raise HTTPException(409, 'partie déjà terminée')

    run_id = s.get('run_id')
    run = _get_run(run_id) if run_id else None

    if not run:
        raise HTTPException(400, 'skip disponible uniquement en mode campagne')

    # Soustraire la moitié des points actuels (arrondi en dessous)
    with _sessions_lock:
        old_points = run['points']
        points_lost = old_points // 2
        run['points'] = old_points - points_lost
        run['current_puzzle_index'] += 1
        s['is_win'] = True  # Marquer comme "complété"

    campaign = s.get('campaign')
    puzzle_id = s['puzzle_id']
    prepared = await _get_prepared_from_request(puzzle_id, campaign)

    return {
        'ok': True,
        'new_score': run['points'],
        'points_lost': points_lost,
        'puzzles_solved': run['puzzles_solved'],
        'win_reveal': _win_reveal_payload(prepared)
    }


# Servir index.html
@app.get("/")
async def serve_index():
    """Serve le client web"""
    from pathlib import Path
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Discoverix API"}


@app.get("/favicon.svg")
async def serve_favicon():
    """Serve le favicon"""
    from pathlib import Path
    favicon_path = Path(__file__).parent / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404)


# Nettoyage périodique
def periodic_cleanup():
    """Nettoyage périodique des sessions"""
    while True:
        time.sleep(3600)
        _cleanup_sessions()


if __name__ == "__main__":
    # Démarrer thread de nettoyage
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()

    # Démarrer serveur
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
