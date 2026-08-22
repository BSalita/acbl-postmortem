"""HTTP client for the ACBL club-results API (src/acbl/acbl_club_api_server.py).

Replaces in-process Playwright scraping in the postmortem app: the API owns
the Chrome profile, throttling, and the club-results cache. Return shapes
mirror mlBridge.mlBridgeAcblLib so downstream app.py code is unchanged:

  player_club_games(pid)   -> {event_id(int): (my_results_url, details_url, msg)}
  session_details_json(id) -> raw details dict or None

Configure with ACBL_CLUB_API_BASE_URL (default http://127.0.0.1:8508).
"""

from __future__ import annotations

import io
import os
from typing import Any, Callable, Dict, Optional, Tuple

import polars as pl
import requests

ACBL_ORIGIN = "https://my.acbl.org"
ACBL_CLUB_API_BASE_URL = os.environ.get(
    "ACBL_CLUB_API_BASE_URL", "http://127.0.0.1:8508"
).rstrip("/")
_TIMEOUT_S = 300  # live Playwright scrapes are slow


class ClubApiClientError(RuntimeError):
    """API-level failure with the server's detail and hint, if any."""

    def __init__(self, detail: str, hint: Optional[str] = None, status_code: Optional[int] = None):
        message = detail if not hint else f"{detail} ({hint})"
        super().__init__(message)
        self.detail = detail
        self.hint = hint
        self.status_code = status_code


def _get_response(
    path: str, params: Optional[Dict[str, Any]] = None
) -> requests.Response:
    url = f"{ACBL_CLUB_API_BASE_URL}{path}"
    try:
        resp = requests.get(
            url,
            params={k: v for k, v in (params or {}).items() if v is not None},
            timeout=_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise ClubApiClientError(
            f"ACBL club API unreachable at {ACBL_CLUB_API_BASE_URL}: {exc}",
            hint="Start it with: python acbl_club_api_server.py (in src/acbl)",
        ) from exc
    if not resp.ok:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        raise ClubApiClientError(
            body.get("detail") or f"{resp.status_code} from {url}",
            hint=body.get("hint"),
            status_code=resp.status_code,
        )
    return resp


def _get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return _get_response(path, params=params).json()


def dataset_info() -> Dict[str, Any]:
    """Unified API health and data-tier metadata."""
    return _get_json("/health")


def player_club_games(
    player_id: str,
    limit: int = 2000,
    prefer_fresh: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[int, Tuple[str, str, str]]]:
    """Club games for one player, shaped like get_club_results_from_acbl_number.

    prefer_fresh=True asks the API to re-scrape my.acbl.org first (matching the
    old always-live behavior) and falls back to the API's cache when the scrape
    fails, e.g. on a cold Cloudflare profile. Returns None for an invalid
    player number and {} when the player has no club games.
    """
    source_url = f"{ACBL_ORIGIN}/club-results/my-results/{player_id}"
    table = None
    if prefer_fresh:
        if progress is not None:
            progress(f"Fetching latest club games from {source_url} ...")
        try:
            table = _get_json(f"/players/{player_id}/games", {"limit": limit, "refresh": True})
        except ClubApiClientError as exc:
            if exc.status_code == 400:
                return None
            if progress is not None:
                progress(
                    f"Fetching latest club games from {source_url} ... "
                    f"error {exc.status_code or 'unknown'}: {exc}.")
            # Scrape failed or nothing live; retry against the cache below.
    if table is None:
        if progress is not None:
            progress("Checking cached and historical club-game listings ...")
        try:
            table = _get_json(f"/players/{player_id}/games", {"limit": limit})
        except ClubApiClientError as exc:
            if exc.status_code == 400:
                return None
            if exc.status_code == 404:
                return {}
            raise

    meta = table.get("meta") or {}
    attempts = meta.get("refresh_attempts") or []
    for attempt in attempts:
        attempt_number = int(attempt.get("attempt") or 1)
        verb = "Fetching" if attempt_number == 1 else "Retrying"
        if attempt.get("status") == "success":
            message = f"{verb} {source_url} ... success."
        else:
            status_code = attempt.get("status_code") or "unknown"
            detail = attempt.get("detail") or "unknown error"
            message = f"{verb} {source_url} ... error {status_code}: {detail}."
        if progress is not None:
            progress(message)
    if progress is not None and meta.get("refresh_failed"):
        progress("Live refresh failed; falling back to local historical data.")
    elif progress is not None and not attempts:
        source = str(meta.get("source") or "local history")
        progress(
            f"Checking cached and historical club-game listings ... "
            f"found in {source}.")

    games: Dict[int, Tuple[str, str, str]] = {}
    for row in table.get("rows", []):
        sid = row.get("session_id")
        try:
            key = int(sid)
        except (TypeError, ValueError):
            continue
        msg = ", ".join(
            str(part)
            for part in (
                sid,
                row.get("date"),
                row.get("club_name"),
                row.get("event"),
                row.get("session"),
                row.get("score"),
            )
            if part is not None
        )
        details_url = row.get("details_url") or f"{ACBL_ORIGIN}/club-results/details/{sid}"
        games[key] = (source_url, details_url, msg)
    # Most recent first, matching the old event-id-descending ordering.
    games = dict(sorted(games.items(), reverse=True))
    return games


def player_tournament_sessions(
    player_id: str,
    limit: int = 2000,
    prefer_fresh: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Tuple[str, str, str, Dict[str, Any]]]]:
    """Tournament sessions from the unified API, shaped for app.py."""
    if progress is not None:
        progress("Fetching latest tournament sessions using the ACBL API ...")
    try:
        table = _get_json(
            f"/tournaments/players/{player_id}/sessions",
            {"limit": limit, "refresh": prefer_fresh},
        )
    except ClubApiClientError as exc:
        if exc.status_code == 400:
            return None
        if progress is not None:
            progress(
                "Fetching latest tournament sessions using the ACBL API "
                f"... error {exc.status_code or 'unknown'}: {exc}.")
        if prefer_fresh:
            table = _get_json(
                f"/tournaments/players/{player_id}/sessions",
                {"limit": limit, "refresh": False},
            )
        else:
            raise
    meta = table.get("meta") or {}
    if progress is not None:
        if meta.get("refresh_failed"):
            progress(
                "Live tournament refresh failed; falling back to API cache "
                "and historical data.")
        else:
            progress(
                "Fetching latest tournament sessions using the ACBL API "
                "... success.")
    source_url = "https://api.acbl.org/v1/tournament/player/history_query"
    sessions: Dict[str, Tuple[str, str, str, Dict[str, Any]]] = {}
    for row in table.get("rows", []):
        sid = str(row.get("session_id") or "")
        if not sid:
            continue
        description = ", ".join(
            str(value)
            for value in (
                row.get("date"),
                row.get("tournament_name"),
                row.get("event_name"),
                row.get("session"),
                row.get("score"),
            )
            if value not in (None, "")
        )
        sessions[sid] = (
            source_url,
            row.get("details_url")
            or f"https://live.acbl.org/event/{sid.replace('-', '/')}/summary",
            description,
            row,
        )
    return sessions


def session_details_json(session_id: Any, refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Raw details JSON, shaped like get_club_results_details_data (dict or None)."""
    data, _source = session_details_json_with_source(session_id, refresh=refresh)
    return data


def session_details_json_with_source(
    session_id: Any, refresh: bool = False
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Raw details JSON and its API-reported provenance label."""
    try:
        resp = _get_response(f"/sessions/{session_id}/raw", {"refresh": refresh})
        return resp.json(), resp.headers.get("X-ACBL-Data-Source")
    except ClubApiClientError as exc:
        if exc.status_code == 404:
            return None, None  # team event or no embedded details JSON
        raise


def session_dataframes(
    session_id: Any, refresh: bool = False
) -> Tuple[Optional[Dict[str, pl.DataFrame]], Optional[str]]:
    """Flat create_club_dfs-compatible frames and their provenance.

    The API serves these from normalized historical parquet when available,
    otherwise from archive/cache/live session JSON.
    """
    try:
        payload = _get_json(f"/sessions/{session_id}/frames", {"refresh": refresh})
    except ClubApiClientError as exc:
        if exc.status_code == 404:
            return None, None
        raise
    tables = payload.get("tables") or {}
    frames = {
        name: pl.DataFrame(rows, strict=False)
        for name, rows in tables.items()
    }
    source = (payload.get("meta") or {}).get("source")
    return frames or None, source


def session_augmented_dataframe(
    session_id: Any,
    player_id: Optional[str] = None,
    refresh: bool = False,
) -> Tuple[Optional[pl.DataFrame], Optional[str]]:
    """Complete API-resolved postmortem, transported as Parquet.

    Resolution is historical augmented parquet, API parquet cache, then a
    headless live build. Streamlit never generates MCP data.
    """
    try:
        resp = _get_response(
            f"/postmortems/{session_id}.parquet",
            {"player_id": player_id, "refresh": refresh},
        )
    except ClubApiClientError as exc:
        if exc.status_code == 404:
            return None, None
        raise
    try:
        frame = pl.read_parquet(io.BytesIO(resp.content))
    except Exception as exc:
        raise ClubApiClientError(
            f"Invalid postmortem parquet returned for session {session_id}: {exc}"
        ) from exc
    return frame, resp.headers.get("X-ACBL-Data-Source")
