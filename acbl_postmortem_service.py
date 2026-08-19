"""Headless access to cached ACBL postmortem dataframes.

The Streamlit app (app.py) persists each fully augmented board-results
dataframe to cache/df-{session_id}-{player_id}.parquet right after
augmentation (see save_augmented_df_to_cache). This module is the shared,
Streamlit-free core used by acbl_postmortem_mcp_server.py: it enumerates
those parquets, re-derives the player personalization columns
(Boards_I_Played, My_Pair, ... -- same logic as change_game_state in app.py),
and runs DuckDB SQL against the dataframe registered as 'self', mirroring how
the app's SQL favorites work.

Env:
  ACBL_POSTMORTEM_CACHE_DIR  cache directory (default ./cache next to this file)
"""

import os
import pathlib
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import polars as pl

_APP_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = pathlib.Path(os.environ.get("ACBL_POSTMORTEM_CACHE_DIR", str(_APP_DIR / "cache")))

CON_REGISTER_NAME = "self"
DEFAULT_SQL_ROW_LIMIT = 500
MAX_SQL_ROW_LIMIT = 2000
MAX_SCHEMA_COLUMNS = 1000

# df-{session_id}-{player_id}.parquet. Tournament session ids contain dashes
# (e.g. 2310369-2801-2) but ACBL player numbers never do, so the player id is
# the trailing dash-free token.
_CACHE_FILE_RE = re.compile(r"^df-(?P<session_id>.+)-(?P<player_id>[^-]+)\.parquet$")

# (player_direction, pair_direction, partner_direction, opponent_pair_direction)
# Same tuples and macro values as change_game_state in app.py.
_SEAT_TUPLES = (
    ("North", "NS", "S", "EW"),
    ("South", "NS", "N", "EW"),
    ("East", "EW", "W", "NS"),
    ("West", "EW", "E", "NS"),
)

# Default column set for the per-board summary tool; intersected with the
# actual dataframe columns since older caches may predate some augmentations.
BOARD_SUMMARY_COLUMNS = [
    "Board", "Contract", "Declarer_Direction", "Declarer_ID", "Declarer_Name",
    "Result", "Tricks", "Score_NS", "Score_EW", "Pct_NS", "Pct_EW",
    "MP_NS", "MP_EW", "MP_Top", "Par_NS", "ParContract",
    "Pair_Number_NS", "Pair_Number_EW", "PBN",
]


def _parse_cache_filename(name: str) -> Optional[Dict[str, str]]:
    m = _CACHE_FILE_RE.match(name)
    if m is None:
        return None
    return {"session_id": m.group("session_id"), "player_id": m.group("player_id")}


def list_cached_postmortems(player_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Cached postmortems (newest file first), optionally for one player."""
    out: List[Dict[str, Any]] = []
    if not CACHE_DIR.is_dir():
        return out
    for f in CACHE_DIR.glob("df-*.parquet"):
        parsed = _parse_cache_filename(f.name)
        if parsed is None:
            continue
        if player_id is not None and parsed["player_id"] != str(player_id):
            continue
        stat = f.stat()
        out.append(
            {
                "player_id": parsed["player_id"],
                "session_id": parsed["session_id"],
                "file": f.name,
                "size_bytes": stat.st_size,
                "cached_at": stat.st_mtime,
            }
        )
    out.sort(key=lambda d: d["cached_at"], reverse=True)
    return out


def dataset_info() -> Dict[str, Any]:
    cached = list_cached_postmortems()
    return {
        "cache_dir": str(CACHE_DIR),
        "cached_postmortems": len(cached),
        "players": sorted({c["player_id"] for c in cached}),
        "note": (
            "Postmortems are produced on demand by the Streamlit app "
            "(https://acbl.postmortem.chat/?player_id=<ACBL number>); this "
            "service reads its parquet cache."
        ),
    }


def _resolve_cache_file(player_id: str, session_id: Optional[str] = None) -> pathlib.Path:
    cached = list_cached_postmortems(player_id)
    if not cached:
        raise FileNotFoundError(
            f"No cached postmortem for player {player_id}. Generate one first by "
            f"loading https://acbl.postmortem.chat/?player_id={player_id} (add "
            f"&session_id=... for a specific game)."
        )
    if session_id is None:
        return CACHE_DIR / cached[0]["file"]  # newest cache file
    for c in cached:
        if c["session_id"] == str(session_id):
            return CACHE_DIR / c["file"]
    raise FileNotFoundError(
        f"No cached postmortem for player {player_id} session {session_id}. "
        f"Cached sessions: {[c['session_id'] for c in cached]}"
    )


# Small in-process cache: one postmortem parquet is ~10^4 rows x ~10^3 columns,
# cheap enough to keep a few resident keyed by (path, mtime).
_df_cache: Dict[Tuple[str, float], pl.DataFrame] = {}
_df_cache_lock = threading.Lock()
_DF_CACHE_MAX = 4


def _read_parquet_cached(path: pathlib.Path) -> pl.DataFrame:
    key = (str(path), path.stat().st_mtime)
    with _df_cache_lock:
        if key in _df_cache:
            return _df_cache[key]
    df = pl.read_parquet(path)
    with _df_cache_lock:
        if len(_df_cache) >= _DF_CACHE_MAX:
            _df_cache.pop(next(iter(_df_cache)))
        _df_cache[key] = df
    return df


def personalize(df: pl.DataFrame, player_id: str) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Add the player-centric flag columns exactly as change_game_state does."""
    pid = str(player_id)
    for player_direction, pair_direction, partner_direction, opponent_pair_direction in _SEAT_TUPLES:
        seat = player_direction[0]
        rows = df.filter(pl.col(f"Player_ID_{seat}").str.contains(pid))
        if rows.height == 0:
            continue
        section_name = rows["section_name"][0]
        pair_number = rows[f"Pair_Number_{pair_direction}"][0]
        partner_id = rows[f"Player_ID_{partner_direction}"][0]
        df = df.with_columns(
            pl.lit(opponent_pair_direction).alias("Opponent_Pair_Direction"),
            (pl.col("section_name") == section_name).alias("My_Section"),
        )
        df = df.with_columns(
            pl.col("My_Section").alias("Our_Section"),
            (pl.col("My_Section") & (pl.col(f"Pair_Number_{pair_direction}") == pair_number)).alias("My_Pair"),
        )
        df = df.with_columns(
            pl.col("My_Pair").alias("Our_Pair"),
            pl.col("My_Pair").alias("Boards_I_Played"),
            pl.col("My_Pair").alias("Boards_We_Played"),
            pl.col("My_Pair").alias("Our_Boards"),
            (pl.col("My_Pair") & (pl.col("Declarer_ID") == pid)).alias("Boards_I_Declared"),
            (pl.col("My_Pair") & (pl.col("Declarer_ID") == partner_id)).alias("Boards_Partner_Declared"),
            (
                pl.col("My_Pair")
                & (
                    (pl.col("Declarer_Direction") == opponent_pair_direction[0])
                    | (pl.col("Declarer_Direction") == opponent_pair_direction[1])
                )
            ).alias("Boards_Opponent_Declared"),
        )
        df = df.with_columns(
            (pl.col("Boards_I_Declared") | pl.col("Boards_Partner_Declared")).alias("Boards_We_Declared"),
        )
        meta = {
            "player_id": pid,
            "player_name": rows[f"Player_Name_{seat}"][0],
            "player_direction": player_direction,
            "partner_id": partner_id,
            "partner_name": rows[f"Player_Name_{partner_direction}"][0],
            "partner_direction": partner_direction,
            "pair_direction": pair_direction,
            "opponent_pair_direction": opponent_pair_direction,
            "section_name": section_name,
            "pair_number": pair_number,
            "game_date": str(df["Date"].first()) if "Date" in df.columns else None,
        }
        return df, meta
    raise ValueError(f"Player {pid} not found in any Player_ID_[NESW] column of the cached postmortem.")


def load_postmortem(player_id: str, session_id: Optional[str] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Load a cached postmortem (latest session when session_id is None) and
    personalize it for the player. Returns (df, meta)."""
    path = _resolve_cache_file(str(player_id), session_id)
    parsed = _parse_cache_filename(path.name)
    df = _read_parquet_cached(path)
    df, meta = personalize(df, str(player_id))
    meta["session_id"] = parsed["session_id"]
    meta["cache_file"] = path.name
    return df, meta


def process_sql_macros(sql: str, meta: Dict[str, Any]) -> str:
    """Same substitutions as PostmortemBase.process_prompt_macros."""
    for macro, key in (
        ("{Player_Direction}", "player_direction"),
        ("{Partner_Direction}", "partner_direction"),
        ("{Pair_Direction}", "pair_direction"),
        ("{Opponent_Pair_Direction}", "opponent_pair_direction"),
    ):
        value = meta.get(key)
        if value is not None:
            sql = sql.replace(macro, str(value))
    return sql


def run_sql(df: pl.DataFrame, sql: str, meta: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
    """Run DuckDB SQL against the postmortem dataframe registered as 'self'."""
    limit = max(1, min(limit or DEFAULT_SQL_ROW_LIMIT, MAX_SQL_ROW_LIMIT))
    sql = process_sql_macros(sql.strip().rstrip(";"), meta)
    # Same convenience as the app's ShowDataFrameTable: allow DuckDB's
    # FROM-first syntax by prepending the table when it is not referenced.
    if f"from {CON_REGISTER_NAME}" not in sql.lower():
        sql = f"FROM {CON_REGISTER_NAME} " + sql
    # external access off: only the registered dataframe is queryable.
    con = duckdb.connect(config={"enable_external_access": "false"})
    try:
        con.register(CON_REGISTER_NAME, df)
        result = con.execute(sql).pl()
    finally:
        con.close()
    truncated = result.height > limit
    result = result.head(limit)
    return {
        "sql": sql,
        "columns": result.columns,
        "rows": result.to_dicts(),
        "row_count": result.height,
        "truncated": truncated,
    }


def board_results(
    df: pl.DataFrame,
    meta: Dict[str, Any],
    only_my_boards: bool = True,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-board rows, defaulting to the boards the player actually played."""
    limit = max(1, min(limit or DEFAULT_SQL_ROW_LIMIT, MAX_SQL_ROW_LIMIT))
    if only_my_boards and "Boards_I_Played" in df.columns:
        df = df.filter(pl.col("Boards_I_Played"))
    wanted = columns or BOARD_SUMMARY_COLUMNS
    missing = [c for c in wanted if c not in df.columns]
    selected = [c for c in wanted if c in df.columns]
    if not selected:
        raise ValueError(f"None of the requested columns exist. Missing: {missing}")
    if "Board" in df.columns:
        df = df.sort("Board")
    df = df.select(selected).head(limit)
    return {
        "meta": meta,
        "columns": selected,
        "missing_columns": missing,
        "rows": df.to_dicts(),
        "row_count": df.height,
    }


def schema_columns(df: pl.DataFrame, pattern: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Column names (with dtypes) of the augmented postmortem dataframe,
    optionally filtered by a case-insensitive regex. The frame has thousands
    of columns, hence the cap."""
    limit = max(1, min(limit or MAX_SCHEMA_COLUMNS, MAX_SCHEMA_COLUMNS))
    names = sorted(df.columns)
    if pattern:
        rx = re.compile(pattern, re.IGNORECASE)
        names = [c for c in names if rx.search(c)]
    truncated = len(names) > limit
    names = names[:limit]
    dtypes = dict(zip(df.columns, (str(t) for t in df.dtypes)))
    return {
        "total_columns": df.width,
        "matched_columns": len(names),
        "truncated": truncated,
        "columns": {c: dtypes[c] for c in names},
    }
