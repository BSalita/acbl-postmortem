"""Pre-game ACBL predictions for one session frame.

Joins player and pair Elo ratings, then predicts Declarer_Direction, Contract,
and Pct_NS with the saved torch models. Shared by the Streamlit app and the
headless postmortem API, so both report the same predictions.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import pathlib
import sys
from typing import Dict, Tuple

import polars as pl

_ROOT = pathlib.Path(__file__).resolve().parent
_MLBRIDGE_PATH = next(
    (path for path in (_ROOT / "mlBridge", _ROOT.parent / "mlBridge") if path.is_dir()),
    None,
)
if _MLBRIDGE_PATH is None:
    raise FileNotFoundError("mlBridge not found at ./mlBridge or ../mlBridge")
if str(_MLBRIDGE_PATH.parent) not in sys.path:
    sys.path.append(str(_MLBRIDGE_PATH.parent))

from mlBridge import mlBridgeAiLib  # noqa: E402
from mlBridge.ffbridge_prediction_features import (  # noqa: E402
    ACBL_DECLARER_WEIGHT,
    attach_room_counts,
    joint_contract_and_declarer,
)

logger = logging.getLogger(__name__)

PREDICTION_TARGETS = ("Declarer_Direction", "Contract", "Pct_NS")


def _player_id_key(column: str) -> pl.Expr:
    """Player ids as stripped strings: parquet stores i32, API frames strings."""
    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.replace(r"\.0$", "")
    )


@functools.lru_cache(maxsize=4)
def _player_elo(path: str, _mtime: float) -> pl.DataFrame:
    # Keep the highest Elo_N when a player has several records on one date.
    return (
        pl.read_parquet(path)
        .with_columns(_player_id_key("Player_ID").alias("Player_ID"))
        .sort(["Date", "Player_ID", "Elo_N"], descending=[False, False, True])
        .unique(subset=["Date", "Player_ID"], keep="first", maintain_order=True)
        .select(["Date", "Player_ID", "Elo_N", "Elo_R_EventStart", "MasterPoints"])
        .with_columns(pl.col("Date").cast(pl.Date))
    )


@functools.lru_cache(maxsize=4)
def _pair_elo(path: str, _mtime: float) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .with_columns(
            pl.col("Pair_IDs").str.split("-").list.sort().list.join("-").alias("Pair_IDs_Sorted_Str")
        )
        .sort(["Date", "Pair_IDs_Sorted_Str", "Elo_N"], descending=[False, False, True])
        .unique(subset=["Date", "Pair_IDs_Sorted_Str"], keep="first", maintain_order=True)
        .select(["Date", "Pair_IDs_Sorted_Str", "Elo_N", "Elo_R_EventStart"])
        .with_columns(pl.col("Date").cast(pl.Date))
    )


ELO_DIR_ENV = "ACBL_ELO_DIR"


def elo_dir() -> pathlib.Path:
    """Folder with the current acbl_*_player/pair_elo_ratings.parquet files.

    The same files the ACBL Elo API reads (in prod, /data). Kept apart from
    SavedModels, which holds only the prediction models, so predictions
    always use the latest Elo build instead of a stale copy.
    """
    value = os.getenv(ELO_DIR_ENV, "").strip()
    if not value:
        raise RuntimeError(f"{ELO_DIR_ENV} is not set; point it at the folder with the ACBL Elo rating parquets")
    return pathlib.Path(value)


def _elo_file(name: str) -> Tuple[str, float]:
    path = elo_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"ACBL Elo ratings not found: {path}")
    return str(path), path.stat().st_mtime


def warm_elo_tables() -> None:
    """Load the club and tournament rating tables into the lru caches.

    Loading and sorting them takes ~5 s; the API calls this in a background
    thread at startup so the first prediction after a restart does not.
    """
    for kind in ("club", "tournament"):
        for name, loader in (("player", _player_elo), ("pair", _pair_elo)):
            path = elo_dir() / f"acbl_{kind}_{name}_elo_ratings.parquet"
            if path.is_file():
                loader(*_elo_file(path.name))


def _join_elo(df: pl.DataFrame, club_or_tournament: str) -> pl.DataFrame:
    if df.select(pl.col("^Elo_.*$")).width:
        raise ValueError(f"Session frame already has Elo columns: {df.select(pl.col('^Elo_.*$')).columns}")
    player_elo = _player_elo(*_elo_file(f"acbl_{club_or_tournament}_player_elo_ratings.parquet"))
    pair_elo = _pair_elo(*_elo_file(f"acbl_{club_or_tournament}_pair_elo_ratings.parquet"))
    initial_rows = df.height
    df = df.with_columns(
        [_player_id_key(f"Player_ID_{direction}").alias(f"Player_ID_{direction}") for direction in "NESW"]
    )
    # The rating tables hold millions of rows; join only this session's
    # players and pairs (filter keeps the Date order join_asof needs).
    session_players = pl.concat([df[f"Player_ID_{direction}"] for direction in "NESW"]).unique()
    player_elo = player_elo.filter(pl.col("Player_ID").is_in(session_players.implode()))
    session_pairs = pl.concat([
        df[f"Pair_IDs_{pair}"].cast(pl.List(pl.String)).list.sort().list.join("-") for pair in ("NS", "EW")
    ]).unique()
    pair_elo = pair_elo.filter(pl.col("Pair_IDs_Sorted_Str").is_in(session_pairs.implode()))
    # join_asof takes the latest rating on or before the game date.
    for direction in "NESW":
        # Augmented frames already carry MasterPoints_*; joining the same name
        # again would make Polars suffix it *_right.
        masterpoints_col = f"MasterPoints_{direction}"
        masterpoints_from_elo = f"{masterpoints_col}__from_elo"
        masterpoints_name = masterpoints_from_elo if masterpoints_col in df.columns else masterpoints_col
        elo_cols = player_elo.rename({
            "Player_ID": f"Player_ID_{direction}",
            "Elo_R_EventStart": f"Elo_R_{direction}_EventStart",
            "Elo_N": f"Elo_N_{direction}",
            "MasterPoints": masterpoints_name,
        })
        df = df.join_asof(elo_cols, left_on="Date", right_on="Date", by=f"Player_ID_{direction}", strategy="backward")
        if masterpoints_from_elo in df.columns:
            df = df.with_columns(
                pl.coalesce(pl.col(masterpoints_col), pl.col(masterpoints_from_elo)).alias(masterpoints_col)
            ).drop(masterpoints_from_elo)
    for pair in ("NS", "EW"):
        elo_cols = pair_elo.rename({
            "Elo_R_EventStart": f"Elo_R_{pair}_EventStart",
            "Elo_N": f"Elo_N_{pair}",
        })
        # Pair_IDs_* is a list of i32 in historical parquet and of strings in
        # some API frames; the Elo key is the ids sorted as text.
        df = df.with_columns(
            pl.col(f"Pair_IDs_{pair}")
            .cast(pl.List(pl.String))
            .list.sort()
            .list.join("-")
            .alias(f"Pair_IDs_{pair}_Sorted_Str")
        )
        df = df.join_asof(
            elo_cols,
            left_on="Date",
            right_on="Date",
            by_left=f"Pair_IDs_{pair}_Sorted_Str",
            by_right="Pair_IDs_Sorted_Str",
            strategy="backward",
        ).drop(f"Pair_IDs_{pair}_Sorted_Str")
    right_cols = df.select(pl.col("^.*_right$")).columns
    if right_cols:
        raise ValueError(f"Elo join produced duplicate columns: {right_cols}")
    if df.height != initial_rows:
        raise ValueError(f"Elo join changed the row count from {initial_rows} to {df.height}")
    return df


def _recenter_pct_ns_pred(df: pl.DataFrame, y_ns: str) -> pl.DataFrame:
    """Shift Pct_NS_Pred so its mean is 0.5 after clipping to [0, 1]."""
    pred_col = f"{y_ns}_Pred"
    target_mean = 0.5
    shift = target_mean - float(df[pred_col].mean())
    for _ in range(10):
        shifted = (pl.col(pred_col) + shift).clip(0.0, 1.0)
        actual_mean = float(df.select(shifted.mean()).item())
        if abs(actual_mean - target_mean) < 1e-6:
            break
        shift += target_mean - actual_mean
    return df.with_columns((pl.col(pred_col) + shift).clip(0.0, 1.0).alias(pred_col))


def _model_input(df: pl.DataFrame, schema: Dict) -> pl.DataFrame:
    to_pl_dtype = mlBridgeAiLib.polars_dtype_from_str
    feature_dtypes = schema.get("feature_dtypes", {})
    required = schema.get("numerical_feature_cols", []) + schema.get("categorical_feature_cols", [])
    # Features the session frame lacks are all-null, as at training time.
    # One schema lookup: df.columns and df[col] rebuild per call, and the
    # classifiers have thousands of features on a ~6,800-column frame.
    present = df.schema
    missing = [col for col in required if col not in present]
    if missing:
        df = df.with_columns([
            pl.lit(None).cast(to_pl_dtype(str(feature_dtypes[col]))).alias(col) for col in missing
        ])
    dtypes = df.schema
    casts = []
    for col in required:
        expected = to_pl_dtype(str(feature_dtypes[col]))
        if dtypes[col] != expected:
            casts.append(pl.col(col).cast(expected, strict=False).alias(col))
    return df.with_columns(casts) if casts else df


def predict_game_results(
    df: pl.DataFrame,
    club_or_tournament: str,
    saved_models_path: pathlib.Path,
) -> pl.DataFrame:
    """Add Elo, *_Pred, *_Actual, and *_Pred_Error columns to one session frame.

    Board_Tables, Contract_Pred_Tables, and Declarer_Direction_Pred_Tables
    count the tables on the deal that played the predicted contract and seat.
    """
    if club_or_tournament not in ("club", "tournament"):
        raise ValueError(f"club_or_tournament must be club or tournament, not {club_or_tournament!r}")
    saved_models_path = pathlib.Path(saved_models_path)
    df = _join_elo(df, club_or_tournament)

    # Live API builds omit game_date. Historical frames carry it as Date at
    # midnight, and a null scales to an extreme input rather than a neutral one.
    if "Date" in df.columns and ("game_date" not in df.columns or df["game_date"].is_null().all()):
        df = df.with_columns(pl.col("Date").cast(pl.Datetime("us")).alias("game_date"))

    class_probabilities = {}
    for y_name in PREDICTION_TARGETS:
        model_name = f"acbl_{club_or_tournament}_predicted_{y_name.lower()}_torch_model"
        schema_path = saved_models_path / f"{model_name}_schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(f"ACBL prediction model schema not found: {schema_path}")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        df = _model_input(df, schema)
        pred_df = mlBridgeAiLib.predict_model(
            saved_models_path,
            model_name,
            df,
            top_k=5 if y_name == "Contract" else 1,
            return_probs=y_name in ("Contract", "Declarer_Direction"),
        )
        if y_name in ("Contract", "Declarer_Direction"):
            class_to_idx = schema["class_to_idx"]
            classes = sorted(class_to_idx, key=class_to_idx.get)
            class_probabilities[y_name] = (
                classes,
                pred_df[f"{y_name}_Probs"].list.to_array(len(classes)).to_numpy(),
            )
        overlapping = [c for c in pred_df.columns if c in df.columns]
        if overlapping:
            df = df.drop(overlapping)
        df = df.hstack(pred_df)

    y_ns, y_ew = "Pct_NS", "Pct_EW"
    df = df.with_columns([
        pl.col(y_ns).alias(f"{y_ns}_Actual"),
        (1 - pl.col(y_ns)).alias(y_ew),
        (1 - pl.col(y_ns)).alias(f"{y_ew}_Actual"),
    ])
    # Recenter within each game so every game keeps a mean of 0.5.
    session_col = next(
        (col for col in ("section_id", "Section_ID", "event_id", "Event_ID", "session_id", "Session_ID")
         if col in df.columns),
        None,
    )
    if session_col and df[session_col].n_unique() > 1:
        df = pl.concat([
            _recenter_pct_ns_pred(df.filter(pl.col(session_col) == value), y_ns)
            for value in df[session_col].unique().sort()
        ])
    else:
        df = _recenter_pct_ns_pred(df, y_ns)
    df = df.with_columns((1.0 - pl.col(f"{y_ns}_Pred")).alias(f"{y_ew}_Pred"))
    df = df.with_columns([
        (pl.col(y_ns) - pl.col(f"{y_ns}_Pred")).alias(f"{y_ns}_Pred_Error"),
        (pl.col(y_ns) - pl.col(f"{y_ns}_Pred")).abs().alias(f"{y_ns}_Pred_Absolute_Error"),
        (pl.col(y_ew) - pl.col(f"{y_ew}_Pred")).alias(f"{y_ew}_Pred_Error"),
        (pl.col(y_ew) - pl.col(f"{y_ew}_Pred")).abs().alias(f"{y_ew}_Pred_Absolute_Error"),
    ])

    # One contract and its declarer from both models, so they always agree.
    # A passed-out board has no declarer (null, as in Declarer_Direction).
    contracts, directions = joint_contract_and_declarer(
        *class_probabilities["Contract"], *class_probabilities["Declarer_Direction"],
        declarer_weight=ACBL_DECLARER_WEIGHT,
    )
    df = df.with_columns([
        pl.Series("Contract_Pred", contracts, dtype=pl.String),
        pl.Series("Declarer_Direction_Pred", [None if d == "NULL" else d for d in directions], dtype=pl.String),
    ])
    # One frame can hold games that reuse a board number for different deals.
    df = attach_room_counts(df, ("Board", "PBN"))
    pred = pl.col("Declarer_Direction_Pred")
    df = df.with_columns([
        pl.coalesce([pl.when(pred == seat).then(pl.col(f"Player_ID_{seat}").cast(pl.String)) for seat in "NESW"])
        .alias("Declarer_ID_Pred"),
        pl.coalesce([pl.when(pred == seat).then(pl.col(f"Player_Name_{seat}").cast(pl.String)) for seat in "NESW"])
        .alias("Declarer_Name_Pred"),
    ])
    logger.info(
        "ACBL %s predictions: %d rows, Pct_NS_Pred mean %.4f",
        club_or_tournament, df.height, float(df["Pct_NS_Pred"].mean()),
    )
    return df
