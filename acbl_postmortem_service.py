"""Headless ACBL postmortem analysis backed only by the unified Results API."""

import pathlib
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import polars as pl

import acbl_club_api_client as club_api

SAVED_MODELS_PATH = pathlib.Path(__file__).resolve().parent / "SavedModels"
# Columns added by acbl_session_predictions; a query naming one needs them.
_PREDICTION_COLUMN_RE = re.compile(r"_Pred\b|_Pred_|_Actual\b|\bElo_", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_PREDICTED_FRAMES: "OrderedDict[Tuple[str, int, Optional[float]], pl.DataFrame]" = OrderedDict()
_PREDICTED_FRAMES_MAX = 8
_PREDICTION_LOCK = threading.Lock()

CON_REGISTER_NAME = "self"
DUCKDB_THREADS = 8
DEFAULT_SQL_ROW_LIMIT = 500
MAX_SQL_ROW_LIMIT = 2000
MAX_SCHEMA_COLUMNS = 1000

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
# TODO: ParContract (the contract text) is not produced by the ACBL
# augmentation pipeline. ParContracts is a list of equally optimal contracts
# and is not a substitute. Par_Contract_NS and Par_Contract_EW are signed
# scores (+1 when that side's DD score is at least par, otherwise -1), not
# the contract itself.
BOARD_SUMMARY_COLUMNS = [
    "Board", "Dealer", "Vul", "Contract", "Declarer_Direction", "Declarer_ID", "Declarer_Name",
    "Result", "Tricks", "Score_NS", "Score_EW", "Pct_NS", "Pct_EW",
    "MP_NS", "MP_EW", "MP_Top", "Par_NS", "ParContract",
    "Par_Contract_NS", "Par_Contract_EW", "Is_Sacrifice_Opportunity",
    "DD_Score_NS", "DD_Score_EW", "EV_Score_NS", "EV_Score_EW",
    "Pair_Number_NS", "Pair_Number_EW", "PBN",
]


def list_available_postmortems(player_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Club and tournament sessions listed by the unified ACBL API."""
    if player_id is None:
        return []
    by_session: Dict[str, Dict[str, Any]] = {}
    try:
        club_games = club_api.player_club_games(
            str(player_id), limit=2000, prefer_fresh=True)
    except club_api.ClubApiClientError:
        club_games = {}
    for session_id, (_source_url, details_url, description) in (
        club_games or {}
    ).items():
        sid = str(session_id)
        by_session[sid] = {
            "player_id": str(player_id),
            "session_id": sid,
            "kind": "club",
            "source": "unified ACBL API",
            "description": description,
            "details_url": details_url,
        }
    try:
        tournament_sessions = club_api.player_tournament_sessions(
            str(player_id), limit=2000, prefer_fresh=True)
    except club_api.ClubApiClientError:
        tournament_sessions = {}
    for sid, (_source_url, details_url, description, row) in (
        tournament_sessions or {}
    ).items():
        by_session[str(sid)] = {
            "player_id": str(player_id),
            "session_id": str(sid),
            "kind": "tournament",
            "source": "unified ACBL API",
            "date": row.get("date"),
            "description": description,
            "details_url": details_url,
        }

    # Club descriptions read "session_id, YYYY-MM-DD, club, ...".
    def sort_key(row: Dict[str, Any]) -> str:
        if row.get("date"):
            return str(row["date"])[:10]
        match = _ISO_DATE_RE.search(str(row.get("description") or ""))
        return match.group(0) if match else ""

    return sorted(by_session.values(), key=sort_key, reverse=True)


def dataset_info() -> Dict[str, Any]:
    info = club_api.dataset_info()
    info["note"] = (
        "All sessions and postmortems come through the unified ACBL API. "
        "The MCP never reads or generates Streamlit caches.")
    return info


def _normalize_player_id_value(value: Any) -> str:
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text


def _player_id_eq(column: str, value: Any) -> pl.Expr:
    """Compare a player-id column to a value regardless of i32 vs string dtype."""
    return (
        pl.col(column)
        .cast(pl.String)
        .str.strip_chars()
        .str.replace(r'\.0$', '')
        == _normalize_player_id_value(value)
    )


def personalize(df: pl.DataFrame, player_id: str) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Add the player-centric flag columns exactly as change_game_state does."""
    pid = _normalize_player_id_value(player_id)
    for player_direction, pair_direction, partner_direction, opponent_pair_direction in _SEAT_TUPLES:
        seat = player_direction[0]
        rows = df.filter(_player_id_eq(f"Player_ID_{seat}", pid))
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
            (pl.col("My_Pair") & _player_id_eq("Declarer_ID", pid)).alias("Boards_I_Declared"),
            (pl.col("My_Pair") & _player_id_eq("Declarer_ID", partner_id)).alias("Boards_Partner_Declared"),
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


def needs_predictions(sql: str) -> bool:
    return _PREDICTION_COLUMN_RE.search(sql) is not None


def session_kind(session_id: str) -> str:
    """Same rule as the unified API: club session ids are all digits."""
    return "club" if str(session_id).isdigit() else "tournament"


def _with_predictions(df: pl.DataFrame, session_id: str) -> pl.DataFrame:
    import acbl_session_predictions as predictions

    # Live sessions change as results arrive, so the key includes the scores.
    key = (
        str(session_id),
        df.height,
        df.select(pl.col("Pct_NS").cast(pl.Float64).sum()).item() if "Pct_NS" in df.columns else None,
    )
    with _PREDICTION_LOCK:
        cached = _PREDICTED_FRAMES.get(key)
        if cached is None:
            cached = predictions.predict_game_results(df, session_kind(session_id), SAVED_MODELS_PATH)
            _PREDICTED_FRAMES[key] = cached
            while len(_PREDICTED_FRAMES) > _PREDICTED_FRAMES_MAX:
                _PREDICTED_FRAMES.popitem(last=False)
        else:
            _PREDICTED_FRAMES.move_to_end(key)
    return cached


def load_postmortem(
    player_id: str,
    session_id: Optional[str] = None,
    predictions: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Load a postmortem exclusively through the unified ACBL API.

    predictions adds the Elo, *_Pred, *_Actual, and *_Pred_Error columns.
    """
    if session_id is None:
        sessions = list_available_postmortems(str(player_id))
        if not sessions:
            raise FileNotFoundError(
                f"No ACBL sessions found for player {player_id}")
        session_id = sessions[0]["session_id"]
    df, source = club_api.session_augmented_dataframe(
        session_id, player_id=str(player_id))
    if df is None:
        raise FileNotFoundError(
            f"Postmortem data is unavailable for session {session_id}")
    if predictions:
        df = _with_predictions(df, str(session_id))
    df, meta = personalize(df, str(player_id))
    meta["session_id"] = str(session_id)
    meta["source"] = source or "unified ACBL API"
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
    con = duckdb.connect(
        config={"enable_external_access": "false", "threads": DUCKDB_THREADS}
    )
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
