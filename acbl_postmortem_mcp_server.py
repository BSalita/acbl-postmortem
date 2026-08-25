"""MCP server exposing ACBL bridge game postmortem data.

Transport: streamable HTTP (endpoint /mcp) on ACBL_POSTMORTEM_MCP_PORT
(default 8511), stateless with JSON responses so plain HTTP clients and
cloudflared work without session affinity. Same pattern as
Elo_Ratings/elo_mcp_server.py.

Every tool calls the first-party ACBL Postmortem REST API. This MCP process
does not import report libraries, read parquet, or call third-party APIs.

Deployment: acbl-postmortem-mcp container, started by
../7nt/postmortem_start.ps1. GET /health is used by the wslc watchdog and
deploy health checks.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

import requests

ACBL_POSTMORTEM_MCP_PORT = int(os.environ.get("ACBL_POSTMORTEM_MCP_PORT", "8511"))
ACBL_POSTMORTEM_API_BASE_URL = os.environ.get(
    "ACBL_POSTMORTEM_API_BASE_URL", "http://127.0.0.1:8522"
).rstrip("/")
_TIMEOUT_S = 300

mcp = MCPServer("acbl-postmortem")


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = requests.get(
        f"{ACBL_POSTMORTEM_API_BASE_URL}{path}",
        params={key: value for key, value in (params or {}).items() if value is not None},
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{ACBL_POSTMORTEM_API_BASE_URL}{path}",
        json=payload,
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness probe for the wslc watchdog / deploy health check."""
    return JSONResponse({"service": "acbl-postmortem-mcp", "api": _get("/health")})


@mcp.tool()
def acbl_postmortem_dataset_info() -> Dict[str, Any]:
    """Summary of unified ACBL API postmortem availability and data tiers."""
    return _get("/acbl-postmortem/dataset-info")


@mcp.tool()
def acbl_postmortem_sessions(player_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """List available ACBL postmortem sessions, optionally for one player.

    For a player, this combines generated caches with club games listed in the
    normalized historical parquet. The boards/sql/schema tools load requested
    historical club sessions directly from the augmented parquet.
    """
    return _get(
        "/acbl-postmortem/sessions",
        {"player_id": player_id, "limit": max(1, min(limit, 500))},
    )


@mcp.tool()
def acbl_postmortem_boards(
    player_id: str,
    session_id: Optional[str] = None,
    only_my_boards: bool = True,
    columns: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Per-board results for one ACBL postmortem: contract, declarer,
    result, tricks, scores, matchpoint percentages, and the deal (PBN).

    session_id: historical club IDs are loaded directly from augmented parquet;
    omit for the player's most recently generated cache.
    only_my_boards: True (default) limits rows to boards the player's pair
    actually played; False returns every board in the game (all pairs).
    columns: optional comma-separated column names to override the default
    summary set (discover names with acbl_postmortem_schema).
    """
    return _get(
        "/acbl-postmortem/boards",
        {
            "player_id": player_id,
            "session_id": session_id,
            "only_my_boards": only_my_boards,
            "columns": columns,
            "limit": limit,
        },
    )


@mcp.tool()
def acbl_postmortem_sql(
    player_id: str,
    sql: str,
    session_id: Optional[str] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    """Run a DuckDB SQL query against one ACBL postmortem, registered
    as table 'self' (one row per board result, thousands of augmented columns:
    double-dummy, par, single-dummy expected values, Elo, HCP, ...).

    'FROM self' is prepended when the query does not reference it, so both
    'SELECT ... FROM self ...' and DuckDB's 'SELECT ...' shorthand work.
    Personalization macros are substituted before execution:
    {Player_Direction}, {Partner_Direction}, {Pair_Direction},
    {Opponent_Pair_Direction}. Boolean helper columns include Boards_I_Played,
    Boards_I_Declared, Boards_We_Declared, Boards_Opponent_Declared, My_Pair.
    session_id: historical club IDs are loaded directly from augmented parquet;
    omit for the player's most recently generated cache.
    """
    return _post(
        "/acbl-postmortem/sql",
        {
            "player_id": player_id,
            "sql": sql,
            "session_id": session_id,
            "limit": limit,
        },
    )


@mcp.tool()
def acbl_postmortem_schema(
    player_id: str,
    session_id: Optional[str] = None,
    pattern: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Column names and dtypes of one ACBL postmortem dataframe. The
    frame has thousands of augmented columns, so pass a case-insensitive regex
    pattern (e.g. 'Pct|Score', '^DD_', 'Elo') to search for relevant ones
    before writing acbl_postmortem_sql queries."""
    return _get(
        "/acbl-postmortem/schema",
        {
            "player_id": player_id,
            "session_id": session_id,
            "pattern": pattern,
            "limit": limit,
        },
    )


if __name__ == "__main__":
    print(
        f"[acbl-postmortem-mcp] start {datetime.now(timezone.utc).isoformat()} "
        f"on :{ACBL_POSTMORTEM_MCP_PORT}; api -> {ACBL_POSTMORTEM_API_BASE_URL}",
        flush=True,
    )
    # Stateless + JSON responses: plain request/response tools, no session
    # affinity needed behind cloudflared, and curl-testable.
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=ACBL_POSTMORTEM_MCP_PORT,
        stateless_http=True,
        json_response=True,
    )
