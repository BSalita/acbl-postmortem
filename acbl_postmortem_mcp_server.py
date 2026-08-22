"""MCP server exposing ACBL bridge game postmortem data.

Transport: streamable HTTP (endpoint /mcp) on ACBL_POSTMORTEM_MCP_PORT
(default 8511), stateless with JSON responses so plain HTTP clients and
cloudflared work without session affinity. Same pattern as
Elo_Ratings/elo_mcp_server.py.

Data sources: the parquet cache written by the Streamlit app and the
pre-augmented historical club parquet exposed by the local ACBL Club API.
Historical cache misses do not scrape ACBL or re-run augmentation.

Deployment: acbl-postmortem-mcp container, started by
../7nt/postmortem_start.ps1. GET /health is used by the wslc watchdog and
deploy health checks.
"""

import os
from typing import Any, Dict, Optional

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

import acbl_postmortem_service as svc

ACBL_POSTMORTEM_MCP_PORT = int(os.environ.get("ACBL_POSTMORTEM_MCP_PORT", "8511"))

mcp = MCPServer("acbl-postmortem")


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness probe for the wslc watchdog / deploy health check."""
    info = svc.dataset_info()
    return JSONResponse(
        {
            "status": "ok",
            "service": "acbl-postmortem-mcp",
            "cache_dir": info["cache_dir"],
            "cached_postmortems": info["cached_postmortems"],
        }
    )


@mcp.tool()
def acbl_postmortem_dataset_info() -> Dict[str, Any]:
    """Summary of cached and historical ACBL postmortem availability."""
    return svc.dataset_info()


@mcp.tool()
def acbl_postmortem_sessions(player_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """List available ACBL postmortem sessions, optionally for one player.

    For a player, this combines generated caches with club games listed in the
    normalized historical parquet. The boards/sql/schema tools load requested
    historical club sessions directly from the augmented parquet.
    """
    sessions = svc.list_available_postmortems(player_id)[: max(1, min(limit, 500))]
    return {"sessions": sessions, "count": len(sessions)}


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
    df, meta = svc.load_postmortem(player_id, session_id)
    cols = [c.strip() for c in columns.split(",")] if columns else None
    return svc.board_results(df, meta, only_my_boards=only_my_boards, columns=cols, limit=limit)


@mcp.tool()
def acbl_postmortem_sql(
    player_id: str,
    sql: str,
    session_id: Optional[str] = None,
    limit: int = svc.DEFAULT_SQL_ROW_LIMIT,
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
    df, meta = svc.load_postmortem(player_id, session_id)
    result = svc.run_sql(df, sql, meta, limit=limit)
    result["meta"] = meta
    return result


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
    df, _ = svc.load_postmortem(player_id, session_id)
    return svc.schema_columns(df, pattern=pattern, limit=limit)


if __name__ == "__main__":
    print(
        f"[acbl-postmortem-mcp] starting on :{ACBL_POSTMORTEM_MCP_PORT} "
        f"(endpoint /mcp, health /health); cache -> {svc.CACHE_DIR}",
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
