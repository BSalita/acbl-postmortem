"""HTTP client for the ACBL club-results API (src/acbl/acbl_club_api_server.py).

Replaces in-process Playwright scraping in the postmortem app: the API owns
the Chrome profile, throttling, and the club-results cache. Return shapes
mirror mlBridge.mlBridgeAcblLib so downstream app.py code is unchanged:

  player_club_games(pid)   -> {event_id(int): (my_results_url, details_url, msg)}
  session_details_json(id) -> raw details dict or None

Configure with ACBL_CLUB_API_BASE_URL (default http://127.0.0.1:8508).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

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


def _get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
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
    return resp.json()


def player_club_games(
    player_id: str,
    limit: int = 2000,
    prefer_fresh: bool = True,
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
        try:
            table = _get_json(f"/players/{player_id}/games", {"limit": limit, "refresh": True})
        except ClubApiClientError as exc:
            if exc.status_code == 400:
                return None
            # Scrape failed or nothing live; retry against the cache below.
    if table is None:
        try:
            table = _get_json(f"/players/{player_id}/games", {"limit": limit})
        except ClubApiClientError as exc:
            if exc.status_code == 400:
                return None
            if exc.status_code == 404:
                return {}
            raise

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
    return dict(sorted(games.items(), reverse=True))


def session_details_json(session_id: Any, refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Raw details JSON, shaped like get_club_results_details_data (dict or None)."""
    try:
        return _get_json(f"/sessions/{session_id}/raw", {"refresh": refresh})
    except ClubApiClientError as exc:
        if exc.status_code == 404:
            return None  # team event or no embedded details JSON
        raise
