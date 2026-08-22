#!/usr/bin/env python
# coding: utf-8

# important: any change to df requires con.register() to be called again

#!pip install openai python-dotenv pandas --quiet

# todo: load_model() is failing if numpy >= 2.0.0 is installed.
# todo: don't automatically report on the most recent game. If the game is errors, it's inconvenient to selecting others.

import logging
from typing import Any, Optional, Dict, List, Tuple, Union
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # or DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def print_to_log_info(*args: Any) -> None:
    print_to_log(logging.INFO, *args)
def print_to_log_debug(*args: Any) -> None:
    print_to_log(logging.DEBUG, *args)
def print_to_log(level: int, *args: Any) -> None:
    logging.log(level, ' '.join(str(arg) for arg in args))

import sys
from collections import defaultdict
import pathlib
#import re
import time
import streamlit as st
import streamlit_chat
from streamlit_extras.bottom_container import bottom
from stqdm import stqdm
#import openai
#from openai import AsyncOpenAI
#from openai import openai_object  # used to suppress vscode type checking errors
import polars as pl
import importlib
import duckdb
import json
import contextlib
import os
from datetime import datetime, timezone
import platform
from dotenv import load_dotenv
#import asyncio
#from streamlit_profiler import Profiler # Profiler -- temp?

def get_db_connection():
    """Get or create a session-specific database connection.
    
    This ensures each Streamlit session has its own database connection,
    preventing concurrency issues when multiple users access the app.
    
    Returns:
        duckdb.DuckDBPyConnection: Session-specific database connection
    """
    if 'db_connection' not in st.session_state:
        # Create a new connection for this session
        st.session_state.db_connection = duckdb.connect()
        print_to_log_info(f"Created new database connection for session")
    return st.session_state.db_connection


@st.cache_data
def load_parquet_cached(file_path: str) -> pl.DataFrame:
    """Load parquet file with Streamlit caching for sharing between users.
    
    This function uses @st.cache_data to load parquet files once and share
    them between all concurrent users, improving memory efficiency and loading speed.
    
    Args:
        file_path: Path to the parquet file as string
        
    Returns:
        pl.DataFrame: Loaded DataFrame
    """
    return pl.read_parquet(file_path)

# Only declared to display version information

import numpy as np
import pandas as pd
#import safetensors
#import sklearn
#import torch
import endplay

# todo: only want to assert if first time. assert os.getenv("ACBL_API_KEY") is None, f"ACBL_API_KEY environment variable should not be set. Remove .streamlit/secrets.toml file? {os.getenv('ACBL_API_KEY')}"
load_dotenv()

# retrieve ACBL API Key
acbl_api_key = os.getenv("ACBL_API_KEY")
assert acbl_api_key is not None, "ACBL_API_KEY environment variable not set. See README.md for instructions."
assert 'Bearer' not in acbl_api_key, "ACBL_API_KEY must not contain 'Bearer' or it will be rejected by ACBL."
assert 'Authorization' not in acbl_api_key, "ACBL_API_KEY must not contain 'Authorization' or it will be rejected by ACBL."

# retrieve OpenAI API Key
#openai_api_key = os.getenv("OPENAI_API_KEY")
#assert openai_api_key is not None, "OPENAI_API_KEY environment variable not set. See README.md for instructions."
#openai_async_client = AsyncOpenAI(api_key=openai_api_key)
#DEFAULT_CHEAP_AI_MODEL = "gpt-3.5-turbo-1106" # -1106 until Dec 11th 2023. "gpt-3.5-turbo" is cheapest. "gpt-4" is most expensive.
#DEFAULT_LARGE_AI_MODEL = "gpt-3.5-turbo-1106" # -1106 until Dec 11th 2023. now cheapest "gpt-3.5-turbo-16k" # might not be needed now that schema size is reduced.
#DEFAULT_AI_MODEL = DEFAULT_LARGE_AI_MODEL
#DEFAULT_GPT4_AI_MODEL = "gpt-4-turbo-preview" # preview is always newest
#DEFAULT_AI_MODEL = DEFAULT_GPT4_AI_MODEL
#DEFAULT_AI_MODEL_TEMPERATURE = 0.0

# todo: doesn't some variation of import chatlib.chatlib work instead of using sys.path.append such as exporting via __init__.py?
#import acbllib.acbllib
#import streamlitlib.streamlitlib
#import chatlib.chatlib
#import mlBridge.mlBridgeLib


# import mlBridge.mlBridgeLib
# import mlBridgeLib.mlBridgeAugmentLib
# import acbllib.acbllib
# import chatlib.chatlib
# import streamlitlib.streamlitlib

# mlBridgeLib.pd_options_display()

_APP_DIR = pathlib.Path(__file__).resolve().parent
_REQUIRED_LIBS = ('mlBridge', 'streamlitlib')

def _is_lib_dir(path: pathlib.Path) -> bool:
    return path.is_dir() and (path / '__init__.py').is_file()

_resolved_libs = []
for _name in _REQUIRED_LIBS:
    _candidates = (
        _APP_DIR / _name,
        _APP_DIR.parent / _name,
        _APP_DIR.parent.parent / _name,
    )
    _found = next((path for path in _candidates if _is_lib_dir(path)), None)
    if _found is None:
        raise FileNotFoundError(
            f"{_name} not found at " + " or ".join(str(path) for path in _candidates)
        )
    _resolved_libs.append(_found)
# Package root for import mlBridge.*; lib dirs first for legacy import streamlitlib/acbllib.
for _p in {_APP_DIR, *(path.parent for path in _resolved_libs)}:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)
for _p in _resolved_libs:
    _s = str(_p)
    if _p.name == 'mlBridge':
        if _s not in sys.path:
            sys.path.append(_s)  # logging_config and friends
    else:
        if _s in sys.path:
            sys.path.remove(_s)
        sys.path.insert(0, _s)

# streamlitlib, mlBridge must be placed after sys.path.append. vscode re-format likes to move them to the top
from mlBridge.mlBridgeAcblLib import (
    get_tournament_sessions_from_acbl_number,
    get_tournament_session_results,
    merge_clean_augment_club_dfs,
    merge_clean_augment_tournament_dfs,
)
# Club-results scraping now lives in the club API (src/acbl/acbl_club_api_server.py);
# this client mirrors the old mlBridgeAcblLib return shapes over HTTP.
import acbl_club_api_client as club_api
import streamlitlib # must be placed after sys.path.append. vscode re-format likes to move this to the top
from mlBridge.mlBridgeLib import pd_options_display, contract_classes, cast_numeric_display_columns # must be placed after sys.path.append. vscode re-format likes to move this to the top
from mlBridge.mlBridgeAugmentLib import (
    AllAugmentations,
)
from mlBridge.mlBridgePostmortemLib import PostmortemBase

# override pandas display options
pd_options_display()

# pd.options.display.float_format = lambda x: f"{x:.2f}" doesn't work with streamlit

def ShowDataFrameTable(df: Any, key: str, query: Optional[str] = None, show_sql_query: bool = True, color_column: Optional[str] = None, tooltips: Optional[Any] = None) -> Optional[Any]:
    if query is None:
        query = f'SELECT * FROM {st.session_state.con_register_name}'
    if show_sql_query and st.session_state.show_sql_query:
        st.text(f"SQL Query: {query}")

    # if query doesn't contain 'FROM self', add 'FROM self ' to the beginning of the query.
    # can't just check for startswith 'from self'. Not universal because 'from self' can appear in subqueries or after JOIN.
    # this syntax makes easy work of adding FROM but is only compatible with DuckDB SQL.
    if f'from {st.session_state.con_register_name}' not in query.lower():
        query = f'FROM {st.session_state.con_register_name} ' + query

    # Choose query engine based on user preference
    query_engine = st.session_state.get('query_engine', 'DuckDB')
    
    try:
        result_df = cast_numeric_display_columns(get_db_connection().execute(query).pl())
        st.text(f"Result is a dataframe of {len(result_df)} rows.")
        if show_sql_query and st.session_state.show_sql_query:
            # Debug: if Pct_NS_Pred exists in result, show stats
            try:
                if 'Pct_NS_Pred' in result_df.columns:
                    stats = result_df.select([
                        pl.col('Pct_NS_Pred').mean().alias('mean'),
                        pl.col('Pct_NS_Pred').std().alias('std'),
                        pl.col('Pct_NS_Pred').min().alias('min'),
                        pl.col('Pct_NS_Pred').max().alias('max'),
                        pl.col('Pct_NS_Pred').n_unique().alias('unique'),
                    ])
                    print(f"DEBUG[APP] ShowDataFrameTable Pct_NS_Pred stats: mean={float(stats[0,'mean']):.6f}, std={float(stats[0,'std']):.6f}, min={float(stats[0,'min']):.6f}, max={float(stats[0,'max']):.6f}, unique={int(stats[0,'unique'])}")
                    print(f"DEBUG[APP] ShowDataFrameTable Pct_NS_Pred sample head: {result_df.select(pl.col('Pct_NS_Pred')).head(5).to_dict(as_series=False)}")
            except Exception:
                pass
        streamlitlib.ShowDataFrameTable(result_df, key) #, color_column=color_column, tooltips=tooltips)
    except Exception as e:
        st.error(f"duckdb exception: error:{e} query:{query}")
        return None
    
    return result_df


def call_create_club_dfs(
    player_id: str, event_url: str
) -> Optional[Dict[str, pl.DataFrame]]:
    session_id = event_url.rstrip('/').split('/')[-1]
    dfs, _source = club_api.session_dataframes(session_id)
    return dfs


# def create_tournament_dfs(player_id, event_url):
#     data = acbllib.get_tournament_results_details_data(event_url, acbl_api_key)
#     if data is None:
#         return None
#     return chatlib.create_tournament_dfs(data)


def create_schema_string(df: Any, con: Any) -> str:

    df_dtypes_d = {}
    dtypes_d = defaultdict(list)
    complex_objects = []

    for col in df.columns:
        assert col not in df_dtypes_d, col
        dtype_name = df[col].dtype
        if dtype_name == pl.Object:
            if isinstance(df[col][0], (list, dict)):
                complex_objects.append(col)
                continue
            dtype_name = pl.String
            df = df.with_columns(pl.col(col).cast(pl.String))
        elif dtype_name == pl.UInt8:
            df = df.with_columns(pl.col(col).cast(pl.Int64))

        df_dtypes_d[col] = dtype_name
        dtypes_d[dtype_name].append(col)

    for obj in complex_objects:
        print_to_log_debug(str(obj), df[obj][0])

    df = df.drop(complex_objects)

    # warning: fake sql CREATE TABLE because types are dtypes not sql types.
    #df_schema_string = f'CREATE TABLE "results" ({",".join([n+" "+t.name for n,t in zip(df.columns,df.dtypes)])})' # using f' not f"
    df_schema_string = 'CREATE TABLE "results" (\n'+',\n'.join(df.columns)+'\n)' # df.sort_values(key=lambda col: col.str.lower())?

    return df_schema_string


def _normalize_session_id_arg(session_id: Any) -> Any:
    """Treat 'latest' (case-insensitive) as None so callers pick the most recent game."""
    if isinstance(session_id, str) and session_id.strip().lower() == 'latest':
        return None
    return session_id


def _result_entry_date(entry: Tuple[Any, ...]) -> datetime:
    """Parse the leading date from a club/tournament result description."""
    description = str(entry[2]) if len(entry) > 2 else ""
    # Club descriptions start with the session id, while tournament
    # descriptions start directly with the date.
    for value in (part.strip() for part in description.split(",")[:2]):
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
    return datetime.min


def _latest_result_entry(
    entries: Dict[Any, Tuple[Any, ...]],
) -> Optional[Tuple[Any, Tuple[Any, ...], datetime]]:
    if not entries:
        return None
    key, entry = max(entries.items(), key=lambda item: _result_entry_date(item[1]))
    return key, entry, _result_entry_date(entry)


def save_augmented_df_to_cache(df: Any, session_id: Any, player_id: str) -> None:
    """Persist the augmented postmortem dataframe for headless consumers
    (acbl_postmortem_mcp_server.py), using the same cache naming as
    ffbridge-postmortem. Write-only by design: the live app always re-augments
    fresh (see the 'no caching' note in change_game_state) so this cache never
    feeds back into the UI."""
    try:
        cache_dir = pathlib.Path('cache')
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f'df-{session_id}-{player_id}.parquet'
        df.write_parquet(cache_file)
        print_to_log_info(f"Saved postmortem cache {cache_file}: shape:{df.shape} size:{cache_file.stat().st_size}")
    except Exception as e:
        print_to_log_info(f"Unable to save postmortem cache for {session_id}-{player_id}: {e}")


def change_game_state(player_id: str, session_id: str) -> None: # todo: rename to session_id?
    global acbl_api_key

    session_id = _normalize_session_id_arg(session_id)

    # Clear prediction cache when loading a new game/session
    st.session_state.predictions_cached = False
    st.session_state.predicted_df = None

    print_to_log_info(f"Retrieving latest results for {player_id}")

    st.markdown('<div style="height: 50px;"><a id="top-of-report" name="top-of-report"></a></div>', unsafe_allow_html=True)

    con = get_db_connection()
    retrieval_status = st.status(
        "Finding the latest club game and tournament session ...",
        expanded=True,
    )

    def report_retrieval(message: str) -> None:
        retrieval_status.write(message)

    t = time.time()
    if player_id in st.session_state.game_urls_d:
        game_urls = st.session_state.game_urls_d[player_id]
        report_retrieval("Using club games already fetched in this browser session.")
    else:
        try:
            game_urls = club_api.player_club_games(
                player_id,
                progress=report_retrieval,
            )
        except club_api.ClubApiClientError as e:
            report_retrieval(f"Club-game lookup failed: {e}.")
            st.error(f"Could not retrieve club games for {player_id}: {e}")
            return False
    if game_urls is None:
        st.error(f"Player number {player_id} not found.")
        return False
    if len(game_urls) == 0:
        st.info(f"No club games found for {player_id}.")
    print_to_log_info('player_club_games time:', time.time()-t) # takes 4s

    with st.spinner(f"Retrieving a list of tournament sessions for {player_id} ..."):
        t = time.time()
        if player_id in st.session_state.tournament_session_urls_d:
            tournament_session_urls = st.session_state.tournament_session_urls_d[player_id]
            report_retrieval(
                "Using tournament sessions already fetched in this browser session.")
        else:
            try:
                tournament_session_urls = get_tournament_sessions_from_acbl_number(
                    player_id, acbl_api_key)
            except Exception as exc:
                report_retrieval(
                    "Fetching latest tournament sessions using the ACBL "
                    f"tournament API ... error: {exc}.")
                st.error(
                    f"Could not retrieve tournament sessions for {player_id}: {exc}")
                return False
        if tournament_session_urls is None:
            report_retrieval(
                "Fetching latest tournament sessions using the ACBL "
                "tournament API ... failed.")
            st.error(f"Player number {player_id} not found.")
            return False
        if len(tournament_session_urls) == 0:
            report_retrieval(
                "Fetching latest tournament sessions using the ACBL "
                "tournament API ... no sessions found.")
            st.info(f"No tournament sessions found for {player_id}.")
        elif player_id not in st.session_state.tournament_session_urls_d:
            report_retrieval(
                "Fetching latest tournament sessions using the ACBL "
                "tournament API ... success.")
        print_to_log_info('get_tournament_sessions_from_acbl_number time:', time.time()-t) # takes 2s
    #tournament_session_urls = {} # just ignore tournament sessions for now

    latest_club = _latest_result_entry(game_urls)
    latest_tournament = _latest_result_entry(tournament_session_urls)
    if latest_club is not None:
        report_retrieval(
            f"Latest club game: {latest_club[1][2]}.")
    if latest_tournament is not None:
        report_retrieval(
            f"Latest tournament session is {latest_tournament[0]}: "
            f"{latest_tournament[1][2]}.")

    if session_id is None:
        if latest_club is not None and (
            latest_tournament is None or latest_club[2] >= latest_tournament[2]
        ):
            session_id = latest_club[0]
            report_retrieval("Club game is latest.")
        elif latest_tournament is not None:
            session_id = latest_tournament[0]
            report_retrieval("Tournament session is latest.")

    # Check if we have any valid games or sessions at all
    has_club_games = game_urls is not None and len(game_urls) > 0
    has_tournament_sessions = tournament_session_urls is not None and len(tournament_session_urls) > 0
    
    if not has_club_games and not has_tournament_sessions:
        st.error(f"No game or tournament sessions found for {player_id}. Please make sure {player_id} is a valid player number.")
        return False

    if session_id is None:
        st.error(f"No game or tournament sessions found for {player_id}. Please make sure {player_id} is a valid player number.")
        return False

    # Normalize session_id type so URL-supplied values (always strings) match the
    # dict keys used by either club games (ints) or tournament sessions (strings).
    if session_id not in game_urls and session_id not in tournament_session_urls:
        try:
            if isinstance(session_id, str):
                alt_session_id = int(session_id)
            else:
                alt_session_id = str(session_id)
            if alt_session_id in game_urls or alt_session_id in tournament_session_urls:
                session_id = alt_session_id
        except (ValueError, TypeError):
            pass

    # A bookmarked URL can retain a session from a different player. The
    # requested player is valid, so fall back to their newest available result
    # instead of incorrectly reporting that the player was not found.
    if session_id not in game_urls and session_id not in tournament_session_urls:
        unavailable_session_id = session_id
        if latest_club is not None and (
            latest_tournament is None or latest_club[2] >= latest_tournament[2]
        ):
            session_id = latest_club[0]
            report_retrieval(
                f"Requested session {unavailable_session_id} is not available "
                f"for player {player_id}; using latest club game {session_id}.")
        elif latest_tournament is not None:
            session_id = latest_tournament[0]
            report_retrieval(
                f"Requested session {unavailable_session_id} is not available "
                f"for player {player_id}; using latest tournament session "
                f"{session_id}.")

    # clear games state aninitialize values which are known to be valid at this point
    reset_game_data() # wipe out all game state data
    st.session_state.player_id = player_id
    st.session_state.game_urls_d[player_id] = game_urls
    st.session_state.tournament_session_urls_d[player_id] = tournament_session_urls

    if session_id in game_urls:
        df = None
        dfs = None
        with st.spinner(f"Collecting data for club game {session_id} and player {player_id}."):
            game_description = game_urls[session_id][2]
            results_url = game_urls[session_id][1]
            t = time.time()
            report_retrieval("Looking for results in local historical data ...")
            try:
                df, details_source = club_api.session_augmented_dataframe(session_id)
            except club_api.ClubApiClientError as e:
                st.error(f"Could not retrieve data for game {session_id}: {e}")
                return False
            if df is not None:
                required = {
                    'event_id', 'session_id', 'section_name', 'Board', 'PBN',
                    'event_type', 'board_scoring_method',
                    'Pair_Number_NS', 'Pair_Number_EW',
                    'Player_ID_N', 'Player_ID_S', 'Player_ID_E', 'Player_ID_W',
                }
                missing = sorted(required.difference(df.columns))
                if missing:
                    st.error(
                        f"Historical postmortem {session_id} is missing required "
                        f"columns: {', '.join(missing)}")
                    return False
                if df['event_type'].drop_nulls().unique().to_list() != ['PAIRS']:
                    st.error(
                        f"Game {session_id} is not an ACBL pairs game. "
                        "Select a different game.")
                    return False
                if df['board_scoring_method'].drop_nulls().unique().to_list() != ['MATCH_POINTS']:
                    st.error(
                        f"Game {session_id} is not a match point game. "
                        "Select a different game.")
                    return False
                if df['PBN'].null_count() == len(df):
                    st.error(
                        f"Game {session_id} has no valid hand records. "
                        "Select a different game.")
                    return False
                report_retrieval(
                    "Results found in local historical augmented data.")
                print_to_log_info(
                    'session_augmented_dataframe time:', time.time()-t,
                    'shape:', df.shape)
            else:
                # The monthly/quarterly monolith does not contain the newest
                # sessions. Build only those from recent JSON/live data.
                report_retrieval(
                    "Results are not available in local historical data.")
                report_retrieval(
                    f"Checking recent local results, then {results_url} ...")
                try:
                    dfs, details_source = club_api.session_dataframes(session_id)
                except club_api.ClubApiClientError as e:
                    st.error(f"Could not retrieve data for game {session_id}: {e}")
                    return False
                if dfs is None or 'event' not in dfs or len(dfs['event']) == 0:
                    st.error(
                        f"Game {session_id} has missing or invalid game data. Must be a Mitchell movement game. Select a different club game or tournament session from left sidebar.")
                    return False
                if details_source == 'live':
                    report_retrieval(
                        f"Fetching results from {results_url} ... success.")
                else:
                    report_retrieval(
                        f"Results found in {details_source or 'recent local data'}.")
                print_to_log_info('dfs:', dfs.keys())

                if dfs['pair_summaries']['pair_number'].n_unique() == 1 or dfs['pair_summaries']['direction'].n_unique() == 1:
                    st.error(
                        f"Game {session_id}. I can only chat about Mitchell movements. Select a different club game or tournament session from left sidebar.")
                    return False
                if dfs['event']['type'][0] != 'PAIRS':
                    st.error(
                        f"Game {session_id} is {dfs['event']['type'][0]}. Expecting an ACBL pairs match point game. Select a different club game or tournament session from left sidebar.")
                    return False
                if dfs['event']['board_scoring_method'][0] != 'MATCH_POINTS':
                    st.error(
                        f"Game {session_id} is {dfs['event']['board_scoring_method'][0]}. Expecting an ACBL pairs match point game. Select a different club game or tournament session from left sidebar.")
                    return False
                if not dfs['sessions']['hand_record_id'][0].isdigit():
                    st.error(
                        f"Game {session_id} has an invalid hand record of {dfs['sessions']['hand_record_id'][0]}. Select a different club game or tournament session from left sidebar.")
                    return False
                print_to_log_info('session_dataframes time:', time.time()-t)

        with st.spinner(f"Processing data for club game: {session_id} and player {player_id}."):
            t = time.time()
            if df is None:
                report_retrieval("Processing results. This will take a minute.")
                assert dfs is not None
                df = merge_clean_augment_club_dfs(dfs, {}, player_id)
                if df is None:
                    st.error(
                        f"Game {session_id} has an invalid game file. Select a different club game or tournament session from left sidebar.")
                    return False
                print_to_log_info('merge_clean_augment_club_dfs time:', time.time()-t)
                df = augment_df(df)
            else:
                report_retrieval("Processing results.")
                print_to_log_info(
                    'Using precomputed historical augmentations for', session_id)
            save_augmented_df_to_cache(df, session_id, player_id)
            with open('df_columns.txt','w') as f:
                for col in sorted(df.columns):
                    f.write(col+'\n')

    elif session_id in tournament_session_urls:
        game_description = tournament_session_urls[session_id][2]
        st.text(f"{game_description}")
        results_url = tournament_session_urls[session_id][1]
        dfs = tournament_session_urls[session_id][3]
        #dfs = create_tournament_dfs(player_id, tournament_session_urls[session_id][3])
        if dfs is None or 'event' not in dfs or len(dfs['event']) == 0:
            st.error(
                f"Session {session_id} has missing or invalid session data. Choose another session.")
            return False
        print_to_log_info(dfs.keys())

        if dfs['event']['game_type'] != 'Pairs':
            st.error(
                f"Session {session_id} is {dfs['event']['game_type']}. Expecting an ACBL pairs match point session. Choose another session.")
            return False

        if dfs['score_score_type'] != 'Matchpoints':
            st.error(
                f"Session {session_id} is {dfs['score_score_type']}. Expecting an ACBL pairs match point session. Choose another session.")
            return False

        with st.spinner(f"Collecting data for tournament {dfs['session']['start_date']} {dfs['session']['description']} session {dfs['session']['id']} number {dfs['session']['session_number']} section {dfs['section']} and player {player_id}."):
            t = time.time()

            report_retrieval(f"Fetching results from {results_url} ...")
            response = get_tournament_session_results(session_id, acbl_api_key)
            if response.status_code != 200:
                report_retrieval(
                    f"Fetching results from {results_url} ... "
                    f"error {response.status_code}.")
                st.error(
                    f"Could not retrieve tournament session {session_id}: "
                    f"HTTP {response.status_code}.")
                return False
            report_retrieval(f"Fetching results from {results_url} ... success.")
            json_results_d = response.json()
            if json_results_d is None:
                st.error(
                    f"Session {session_id} has an invalid tournament session file. Choose another session.")
                return False
            print_to_log_info('json_results_d:',json_results_d.keys())

            if len(json_results_d['sections']) == 0:
                st.error(
                    f"Session {session_id} has no sections. Choose another session.")
                return False

            if 'handrecord' not in json_results_d or len(json_results_d['handrecord']) == 0 or 'box_number' not in json_results_d or not json_results_d['box_number'].isdigit():
                st.error(
                    f"Session {session_id} has a missing hand record. Cannot chat about shuffled sessions. Choose another session.")
                return False

            for section in json_results_d['sections']: # is it better/possible to only examine the section which the player played in?

                if section['scoring_type'] != 'Matchpoints':
                    st.error(
                        f"Session {session_id} section {section['section_label']} is {section['scoring_type']}. Expecting an ACBL pairs match point session. Choose another session.")
                    return False

                if section['movement_type'] != 'Mitchell':
                    st.error(
                        f"Session {session_id} section {section['section_label']} is {section['movement_type']}. I can only chat about Mitchell movements. Choose another session.")
                    return False
            print_to_log_info('get_tournament_session_results time:', time.time()-t)

        with st.spinner(f"Processing data for tournament session {session_id} for player {player_id}."):
            report_retrieval("Processing results. This will take a minute.")
            t = time.time()
            #with Profiler():

            df = merge_clean_augment_tournament_dfs(dfs, json_results_d, acbl_api_key, player_id)
            if df is None:
                st.error(
                    f"Session {session_id} has an invalid tournament session file. Choose another session.")
                return False
            print_to_log_info('merge_clean_augment_tournament_dfs time:', time.time()-t)
            #df = acbllib.convert_ffdf_to_mldf(df)
            df = augment_df(df)
            save_augmented_df_to_cache(df, session_id, player_id)
            with open('df_columns.txt','w') as f:
                for col in sorted(df.columns):
                    f.write(col+'\n')

    else:
        st.error(f"Session {session_id} was not found for player {player_id}.")
        return False

    # No more user errors possible. Everything checks out so it's safe to update the session state with new data.
    #reset_game_data() # wipe out all game state data
    #st.session_state.player_id = player_id
    st.session_state.session_id = session_id
    st.session_state.game_description = game_description
    #st.session_state.game_urls_d[player_id] = game_urls
    st.session_state.game_results_url = results_url
    #st.session_state.tournament_session_urls_d[player_id] = tournament_session_urls
    st.session_state.sql_query_mode = False
    st.session_state.main_section_container = st.container(border=True)

    with st.spinner(f"Creating everything data table."):
        t = time.time()
        assert df is not None
        # List of columns to move to the front of df.
        move_to_front = ['Board', 'Contract', 'Result', 'Tricks', 'Score_NS', 'Pct_NS', 'Par_NS']
        # Reorder columns
        new_column_order = move_to_front + [col for col in df.columns if col not in move_to_front]
        df = df[new_column_order]
        st.session_state.df = df
        st.session_state.matchpoint_ns_d = {} # todo: obsolete this -- matchpoint_ns_d

    # Convert 'Date' to datetime and extract scalers
    assert df['Date'].n_unique() == 1, "Oops. Date is non-unique."
    st.session_state.game_date = df['Date'].first() # showing time in case player played multiple games on same day

    # Iterate over player directions
    for player_direction, pair_direction, partner_direction, opponent_pair_direction in [('North', 'NS', 'S', 'EW'), ('South', 'NS', 'N', 'EW'), ('East', 'EW', 'W', 'NS'), ('West', 'EW', 'E', 'NS')]:
        rows = df.filter(pl.col(f"Player_ID_{player_direction[0]}").str.contains(st.session_state.player_id))
        print(f"{st.session_state.player_id=} {rows.height=}")
        if rows.height > 0:
            st.session_state.player_id = player_id
            st.session_state.player_direction = player_direction
            st.session_state.pair_direction = pair_direction
            st.session_state.partner_direction = partner_direction
            st.session_state.opponent_pair_direction = opponent_pair_direction

            session_ids = rows.select('session_id').unique()
            assert session_ids.height == 1, "Oops. session_id non-unique."
            st.session_state.session_id = session_ids[0, 0]

            section_names = rows.select('section_name').unique()
            assert section_names.height == 1, "Oops. section_name non-unique."
            st.session_state.section_name = section_names[0, 0]

            player_names = rows.select(f"Player_Name_{player_direction[0]}").unique()
            assert player_names.height == 1, "Oops. player_names non-unique."
            st.session_state.player_name = player_names[0, 0]

            pair_numbers = rows.select(f"Pair_Number_{pair_direction}").unique() # todo: pair_numbers have a suffix of 'NS' or 'EW'. rename to pair_id? rename pair_id to section_pair_id?
            assert pair_numbers.height == 1, "Oops. pair_numbers non-unique."
            st.session_state.pair_number = pair_numbers[0, 0]
 
            partner_ids = rows.select(f"Player_ID_{partner_direction}").unique()
            assert partner_ids.height == 1, "Oops. partner_ids non-unique."
            st.session_state.partner_id = partner_ids[0, 0]

            partner_names = rows.select(f"Player_Name_{partner_direction}").unique()
            assert partner_names.height == 1, "Oops. partner_names non-unique."
            st.session_state.partner_name = partner_names[0, 0]

            st.session_state.pair_id = f"{st.session_state.section_name}-{st.session_state.pair_number}"

            # Add new columns based on conditions
            df = df.with_columns([
                pl.lit(st.session_state.opponent_pair_direction).alias('Opponent_Pair_Direction'),
                (pl.col('section_name') == st.session_state.section_name).alias('My_Section'),
            ])
            df = df.with_columns([
                pl.col('My_Section').alias('Our_Section'),
                (pl.col('My_Section') & (pl.col(f"Pair_Number_{st.session_state.pair_direction}") == st.session_state.pair_number)).alias('My_Pair'),
            ])
            df = df.with_columns([
                (pl.col('My_Pair')).alias('Our_Pair'),
                (pl.col('My_Pair')).alias('Boards_I_Played'),
                (pl.col('My_Pair')).alias('Boards_We_Played'),
                (pl.col('My_Pair')).alias('Our_Boards'),
                (pl.col('My_Pair') & (pl.col('Declarer_ID') == st.session_state.player_id)).alias('Boards_I_Declared'),
                (pl.col('My_Pair') & (pl.col('Declarer_ID') == st.session_state.partner_id)).alias('Boards_Partner_Declared'),
                (pl.col('My_Pair') & ((pl.col('Declarer_Direction') == opponent_pair_direction[0]) | (pl.col('Declarer_Direction') == opponent_pair_direction[1]))).alias('Boards_Opponent_Declared')
            ])
            df = df.with_columns([
                (pl.col('Boards_I_Declared') | pl.col('Boards_Partner_Declared')).alias('Boards_We_Declared'),
            ])
            break # only want to do this once.

        print_to_log_info(f"create everything data table: {player_direction=} time:{time.time()-t}")
    assert st.session_state.player_id is not None, f"{st.session_state.player_id=}" # can assert if non-mitchell movement isn't already detected.

    # (debug removed)

    # make predictions only if user pressed the AI Predictions button
    if st.session_state.get('ai_predictions_requested', False):
        with st.spinner(f"Making AI Predictions. Takes 20 seconds."):
            t = time.time()
            df_with_predictions = Predict_Game_Results(df) # returning updated df for con.register()
            if df_with_predictions is not None:
                df = df_with_predictions
            else:
                st.error("AI predictions failed. Cannot continue without predictions.")
                st.stop()
        print_to_log_info('Predict_Game_Results time:', time.time()-t) # takes 10s
        # reset request flag so predictions are only run on explicit button press
        st.session_state.ai_predictions_requested = False
    else:
        print_to_log_info('AI predictions skipped (button not pressed).')

    # (debug removed)

    # Create a DuckDB table from the DataFrame
    # register df as a table named 'self' for duckdb discovery. SQL queries will reference this df/table.
    
    # (debug removed)
    
    # FIX: Ensure clean DuckDB registration by dropping any existing table first
    table_name = st.session_state.con_register_name
    # (debug removed)
    
    # Drop existing table if it exists to prevent schema corruption
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        pass
    except Exception as e:
        pass
    
    assert df.select(pl.col(pl.Object)).is_empty(), f"Found Object columns: {df.select(pl.col(pl.Object)).columns}"
    df = cast_numeric_display_columns(df)
    st.session_state.df = df
    con.register(table_name, df) # ugh, df['scores_l'] must be previously dropped otherwise this hangs. reason unknown.
    
    # (debug removed)
    
    # (debug removed)

    st.session_state.df_schema_string = create_schema_string(df, con)
    # todo: temp for debugging
    with open('df_schema.sql','w') as f:
        f.write(st.session_state.df_schema_string)

    # function_calls = [
    #     {
    #         "name": "ask_database",
    #         # from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
    #         # Needs to explictly be told to return a SQL string otherwise returns json.
    #         "description": "Use this function to answer user questions about duplicate bridge statistics. Output should be a fully formed SQL query.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "query": {
    #                     "type": "string",
    #                     "description": f"""
    #                             SQL query for extracting info to answer the user"s question.
    #                             SQL should be written using this database schema:
    #                             {st.session_state.df_schema_string}
    #                             The schema contains table name, column name and column type.
    #                             The returned value should be a plain text SQL query embedded in JSON.
    #                             """,
    #                 }
    #             },
    #             "required": ["query"],
    #         },
    #     }
    # ]
    # st.session_state.function_calls = function_calls
    # reset_messages()

    #content = slash_about()
    #streamlit_chat.message(f"Morty: {content}", logo=st.session_state.assistant_logo)

    retrieval_status.update(
        label=f"Loaded session {session_id}.",
        state="complete",
        expanded=True,
    )
    return True # no errors


def slash_about() -> str:
    content = f"Hey {st.session_state.player_name} ({st.session_state.player_id}), let's chat about your game on {st.session_state.game_date} (event id {st.session_state.session_id}). Your pair was {st.session_state.pair_number}{st.session_state.pair_direction} in section {st.session_state.section_name}. You played {st.session_state.player_direction}. Your partner was {st.session_state.partner_name} ({st.session_state.partner_id}) who played {st.session_state.partner_direction}."
    return content


# def ask_questions_without_context(ups, model=None):
#     # pandasai doesn't work. context length is limited to 4100 tokens. need 8k?
#     # llm = OpenAI(api_token=openai.api_key)
#     # df = st.session_state.df
#     # sdf = SmartDataframe(df, config={"llm": llm})
#     # #sdf = SmartDataframe(df)
#     # qdf = sdf.chat("Show board, contract")
#     # print_to_log('qdf:', qdf)

#     if model is None:
#         model = st.session_state.ai_api
#     function_calls = st.session_state.function_calls
#     # ups can be a string, list of strings, or list of lists of strings.
#     assert isinstance(ups, list), ups
#     with st.spinner(f"Morty is judging you ..."): # {len(ups)} responses from {model}."):
#         tasks = []
#         list_of_new_messages = []
#         for i, up in enumerate(ups):
#             # pass system prompt only
#             new_messages = [st.session_state.messages[0]]
#             list_of_new_messages.append(new_messages)
#             tasks.append(async_chat_up_user(up, new_messages, function_calls, model))
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         try:
#             results = loop.run_until_complete(asyncio.gather(*tasks))
#             #for result in results:
#             #    print_to_log(result)
#         finally:
#             loop.close()
#         for messages in list_of_new_messages:
#             # append all new messages to list except for system prompt
#             st.session_state.messages.extend(messages[1:])


# def ask_a_question_with_context(ups, model=None):
#     t = time.time()
#     if model is None:
#         model = st.session_state.ai_api
#     messages = st.session_state.messages
#     # removed because no longer an issue?
#     #if model != DEFAULT_LARGE_AI_MODEL:
#     #    if len(messages) > 12:
#     #        messages = messages[:1+3]+messages[1+3-10:]
#     function_calls = st.session_state.function_calls
#     if isinstance(ups, list):
#         for i, up in enumerate(ups):
#             with st.spinner(f"Waiting for response {i} of {len(ups)} from {model}."):
#                 chat_up_user(up, messages, function_calls, model)
#     else:
#         with st.spinner(f"Waiting for response from {model}."):
#             chat_up_user(ups, messages, function_calls, model)
#     st.session_state.messages = messages
#     print_to_log_info('ask_a_question_with_context time:', time.time()-t)


# def reset_messages():
#     assert st.session_state.player_id is not None, "Oops. Player number is None."
#     assert st.session_state.system_prompt is not None, "Oops. System prompt is None."
#     augmented_system_prompt = st.session_state.system_prompt
#     #augmented_system_prompt += f" Player Number is '{st.session_state.player_id}'."
#     #augmented_system_prompt += f" Player Name is '{st.session_state.player_name}'."
#     # todo: howell, individual, team not implemented!
#     #augmented_system_prompt += f" Game Date is always {st.session_state.game_date}."
#     #augmented_system_prompt += f" Event ID is always {st.session_state.session_id}."
#     #augmented_system_prompt += f" My partner's Player Direction is always {st.session_state.partner_direction[0]}."
#     #augmented_system_prompt += f" My Player Direction is always {st.session_state.player_direction[0]}."
#     #augmented_system_prompt += f" My partner's Player Direction is always {st.session_state.partner_direction[0]}."
#     #augmented_system_prompt += f" My partner's Player Number is always '{st.session_state.player_id}'."
#     #augmented_system_prompt += f" My partner's Player Name is always '{st.session_state.partner_name}'."
#     #augmented_system_prompt += f" My Pair Direction is always {st.session_state.pair_direction}."
#     #augmented_system_prompt += f" My 'Pair Number {st.session_state.opponent_pair_direction}' is always {st.session_state.pair_number}."
#     #augmented_system_prompt += f" Opponent Pair Direction is always {st.session_state.opponent_pair_direction}."
#     #augmented_system_prompt += f" My Session Id is always {st.session_state.session_id}."
#     #augmented_system_prompt += f" My Section Name is always {st.session_state.section_name}."
#     #augmented_system_prompt += f" \"My boards means\" boards having a Declarer Number of always {st.session_state.player_id}."
#     #augmented_system_prompt += f" \"Our boards\" means boards having a Pair Number always {st.session_state.pair_direction} of {st.session_state.pair_number}."
#     #augmented_system_prompt += f" \"Boards I declared\" means boards having my Declarer Number of always {st.session_state.player_id}."
#     #augmented_system_prompt += f" \"Boards my partner declared\" means boards having my Declarer Number of always {st.session_state.player_id}"
#     #augmented_system_prompt += f" \"Boards we declared\" means boards having a Declarer Number of always {st.session_state.player_id} of {st.session_state.player_id}."
#     #augmented_system_prompt += f" \"Boards we played\" means boards having a Pair Number always {st.session_state.pair_direction} of {st.session_state.pair_number}."
#     st.session_state.augmented_system_prompt = augmented_system_prompt
#     system_message = {"role": "system",
#                       "content": st.session_state.augmented_system_prompt}
#     messages = [system_message]
#     st.session_state.messages = messages


# ---- URL Query Parameter Sync ----
# Sidebar options below are kept in sync with the browser URL so links are
# shareable/bookmarkable: ?player_id=...&session_id=...|latest&show_sql_query=1&sd_samples=10
_URL_PARAM_KEYS = ('player_id', 'session_id', 'show_sql_query', 'sd_samples')


def _set_query_param(name: str, value: Any) -> None:
    """Set or remove a single URL query parameter to match a session state value."""
    if value is None or value == '':
        if name in st.query_params:
            del st.query_params[name]
    else:
        st.query_params[name] = str(value)


def sync_url_params_from_state() -> None:
    """Write the currently selected sidebar options back to the URL query string."""
    _set_query_param('player_id', st.session_state.get('player_id'))
    _set_query_param('session_id', st.session_state.get('session_id'))
    show_sql = st.session_state.get('show_sql_query')
    _set_query_param(
        'show_sql_query',
        None if show_sql is None else ('1' if show_sql else '0'),
    )
    _set_query_param('sd_samples', st.session_state.get('single_dummy_sample_count'))


def apply_url_params_to_state() -> None:
    """Seed sidebar-related session state from URL query parameters.

    Safe to call multiple times; missing/blank params are ignored. Call after the
    normal session defaults are initialized so URL values take precedence.
    """
    qp = st.query_params

    if 'player_id' in qp:
        pid = qp['player_id']
        if isinstance(pid, str) and pid.strip():
            st.session_state.player_id = pid.strip()

    if 'session_id' in qp:
        sid = qp['session_id']
        if isinstance(sid, str) and sid.strip():
            sid = sid.strip()
            # 'latest' means most recent club/tournament game (same as omitting session_id).
            if sid.lower() == 'latest':
                st.session_state.session_id = None
            else:
                # Club session_ids are ints in game_urls_d; tournament session_ids are
                # strings in tournament_session_urls_d. Prefer int when possible and
                # let change_game_state normalize as needed.
                try:
                    st.session_state.session_id = int(sid)
                except ValueError:
                    st.session_state.session_id = sid

    if 'show_sql_query' in qp:
        v = str(qp['show_sql_query']).strip().lower()
        st.session_state.show_sql_query = v in ('1', 'true', 'yes', 'on')

    if 'sd_samples' in qp:
        try:
            n = int(qp['sd_samples'])
            if 1 <= n <= 100:
                st.session_state.single_dummy_sample_count = n
        except (ValueError, TypeError):
            pass


def player_id_change() -> None:
    # todo: looks like there's some situation where this is not called because player_id_input is already set. Need to breakpoint here to determine why st.session_state.player_id isn't updated.
    # assign changed textbox value (player_id_input) to player_id
    player_id = st.session_state.player_id_input
    
    # If the new player ID is empty or the same as current, don't process
    if not player_id or player_id == st.session_state.get('player_id'):
        return
    
    # Attempt to change game state with new player ID
    # This will load games list and auto-select the first game
    success = change_game_state(player_id, None)
    
    if success:
        # Success - set flag and return (Streamlit will auto-rerun after callback)
        st.session_state.player_id_just_changed = True
        sync_url_params_from_state()
        return
    else:
        # If validation failed, set a flag to handle UI reset in main flow
        st.session_state.player_id_validation_failed = True
        st.session_state.invalid_player_id_input = player_id
        # Don't update st.session_state.player_id - keep the previous working one
        sync_url_params_from_state()



def debug_player_id_names_change() -> None:
    # assign changed selectbox value (debug_player_id_names_selectbox). e.g. ['2663279','Robert Salita']
    player_id_name = st.session_state.debug_player_id_names_selectbox
    #if not chat_initialize(player_id_name[0], None):  # grab player number
    #    chat_initialize(st.session_state.player_id, None)
    success = change_game_state(player_id_name[0], None)
    
    # If validation failed for debug player, set flag to handle UI reset in main flow
    if success is False:
        # Set flag to indicate debug validation failure
        st.session_state.debug_player_id_validation_failed = True
        st.session_state.debug_player_id_names_selectbox = None
    sync_url_params_from_state()


def club_session_id_change() -> None:
    #st.session_state.tournament_session_ids_selectbox = None # clear tournament index whenever club index changes. todo: doesn't seem to update selectbox with new index.
    selection = st.session_state.club_session_ids_selectbox
    if selection is not None:
        session_id = int(selection.split(',')[0]) # split selectbox item on commas. only want first split.
        if not change_game_state(st.session_state.player_id, session_id):
            st.session_state.session_id = None
    sync_url_params_from_state()


def tournament_session_id_change() -> None:
    #st.session_state.club_session_ids_selectbox = None # clear club index whenever tournament index changes. todo: doesn't seem to update selectbox with new index.
    selection = st.session_state.tournament_session_ids_selectbox
    if selection is not None:
        session_id = selection.split(',')[0] # split selectbox item on commas. only want first split.
        if not change_game_state(st.session_state.player_id, session_id):
            st.session_state.session_id = None
    sync_url_params_from_state()


def show_sql_query_change() -> None:
    # toggle whether to show sql query
    st.session_state.show_sql_query = st.session_state.sql_query_checkbox
    sync_url_params_from_state()


# todo: feature removed
# def ai_api_selectbox_change():
#     # assign changed selectbox value (ai_api_selectbox) to ai_api
#     st.session_state.ai_api = st.session_state.ai_api_selectbox


def prompts_selectbox_change() -> None:
    if st.session_state.prompts_selectbox is not None:
        title = st.session_state.prompts_selectbox
        if st.session_state.vetted_prompt_titles is not None: # this fixes the situation when an unsupported game event is selected.
            box = st.session_state.vetted_prompt_titles[title]
            ups = box['prompts']
            # if len(ups):
            #     ask_questions_without_context(ups, st.session_state.ai_api)
            read_configs()


def single_dummy_sample_count_changed() -> None:
    st.session_state.single_dummy_sample_count = st.session_state.single_dummy_sample_count_number_input
    change_game_state(st.session_state.player_id, st.session_state.session_id)
    sync_url_params_from_state()


def chat_input_on_submit() -> None:
    prompt = st.session_state.main_prompt_chat_input
    sql_query = process_prompt_macros(prompt)
    if not st.session_state.sql_query_mode:
        st.session_state.sql_query_mode = True
        st.session_state.sql_queries.clear()
    st.session_state.sql_queries.append((prompt,sql_query))
    st.session_state.main_section_container = st.empty()
    st.session_state.main_section_container = st.container()
    with st.session_state.main_section_container:
        # Only execute queries if we have a player_id and data table is ready
        if st.session_state.player_id is not None and hasattr(st.session_state, 'df') and st.session_state.df is not None:
            try:
                # Check if DuckDB table is actually registered
                con = get_db_connection()
                con.execute(f"DESCRIBE {st.session_state.con_register_name}").fetchall()
                # Table exists, execute queries
                for i, (prompt,sql_query) in enumerate(st.session_state.sql_queries):
                    ShowDataFrameTable(None, query=sql_query, key=f'user_query_main_doit_{i}')
            except Exception:
                st.info(f"Data is loading... Please wait for processing to complete.")
        else:
            st.info("Please enter a player ID to load data and execute queries.")

import mlBridgeAiLib # todo: move to top of file
#import mlBridgeAiLib_obsolete # todo: remove when done

def recenter_percentage_predictions(
    df: pl.DataFrame,
    pred_ns_col: str = 'Pct_NS_Pred',
    pred_ew_col: str = 'Pct_EW_Pred',
    actual_ns_col: str = 'Pct_NS_Actual',
    actual_ew_col: str = 'Pct_EW_Actual',
    target_mean: float = 0.5,
) -> pl.DataFrame:
    """Recenter predicted NS percentages to target_mean, clip to [0,1],
    set EW to complementary, and recompute error metrics.

    If the prediction column is missing or empty, returns df unchanged.
    """
    if pred_ns_col not in df.columns:
        return df
    if df.is_empty():
        return df

    ns_mean_df = df.select(pl.col(pred_ns_col).mean())
    if ns_mean_df.is_empty():
        return df
    ns_mean = ns_mean_df.item()
    if ns_mean is None:
        return df
    shift = target_mean - ns_mean

    out = df.with_columns([
        pl.min_horizontal(1.0, pl.max_horizontal(0.0, pl.col(pred_ns_col).add(pl.lit(shift)))).alias(pred_ns_col),
    ])

    # Ensure EW is complementary and update error columns if actuals exist
    out = out.with_columns([
        pl.lit(1).sub(pl.col(pred_ns_col)).alias(pred_ew_col),
    ])

    # derive dynamic error column names from prediction column bases
    def _base_name(col_name: str) -> str:
        return col_name[:-5] if col_name.endswith('_Pred') else col_name

    ns_base = _base_name(pred_ns_col)
    ew_base = _base_name(pred_ew_col)
    ns_err_col = f"{ns_base}_Pred_Error"
    ns_abs_err_col = f"{ns_base}_Pred_Absolute_Error"
    ew_err_col = f"{ew_base}_Pred_Error"
    ew_abs_err_col = f"{ew_base}_Pred_Absolute_Error"

    if actual_ns_col in out.columns:
        out = out.with_columns([
            pl.col(actual_ns_col).sub(pl.col(pred_ns_col)).alias(ns_err_col),
            pl.col(ns_err_col).abs().alias(ns_abs_err_col),
        ])
    if actual_ew_col in out.columns:
        out = out.with_columns([
            pl.col(actual_ew_col).sub(pl.col(pred_ew_col)).alias(ew_err_col),
            pl.col(ew_err_col).abs().alias(ew_abs_err_col),
        ])

    return out

def recenter_percentage_predictions_for_target(
    df: pl.DataFrame,
    y_name: str,
    target_mean: float = 0.5,
) -> pl.DataFrame:
    """Convenience wrapper using y_name to derive column names.

    Uses columns: f"{y_name}_Pred", f"{y_name}_Actual" and the EW counterpart
    derived by swapping 'NS' and 'EW' in y_name.
    """
    if 'NS' in y_name:
        y_name_ns = y_name
        y_name_ew = y_name.replace('NS', 'EW')
    elif 'EW' in y_name:
        y_name_ew = y_name
        y_name_ns = y_name.replace('EW', 'NS')
    else:
        # Cannot infer pair; return unchanged
        return df

    return recenter_percentage_predictions(
        df,
        pred_ns_col=f"{y_name_ns}_Pred",
        pred_ew_col=f"{y_name_ew}_Pred",
        actual_ns_col=f"{y_name_ns}_Actual",
        actual_ew_col=f"{y_name_ew}_Actual",
        target_mean=target_mean,
    )

def recenter_probability_mean_logit(
    df: pl.DataFrame,
    pred_col: str,
    target_mean: float = 0.5,
) -> pl.DataFrame:
    """Recenter probability column to target mean using a logit-domain shift.

    Preserves distribution shape (std stays approximately the same) better than
    linear shifting. No effect if column missing/empty or already near target.
    """
    if pred_col not in df.columns or df.is_empty():
        return df
    try:
        stats = df.select([
            pl.col(pred_col).mean().alias("mean"),
            pl.col(pred_col).std().alias("std"),
        ])
        mu = float(stats[0, "mean"]) if stats.height else None
        sd = float(stats[0, "std"]) if stats.height else None
        if mu is None or sd is None:
            return df
        if abs(mu - target_mean) < 1e-6 or sd < 1e-9:
            return df

        # Extract column as numpy
        p = df[pred_col].to_numpy()
        p = p.astype(float)
        # Clip to avoid infinities in logit
        eps = 1e-6
        p = np.clip(p, eps, 1.0 - eps)
        l = np.log(p / (1.0 - p))

        # Bisection on logit shift c so that mean(sigmoid(l - c)) == target_mean
        lo, hi = -20.0, 20.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            m = float((1.0 / (1.0 + np.exp(-(l - mid)))).mean())
            if m > target_mean:
                lo = mid
            else:
                hi = mid
        c = (lo + hi) / 2.0
        p_new = 1.0 / (1.0 + np.exp(-(l - c)))
        # Replace column
        out = df.with_columns([
            pl.Series(pred_col, p_new.astype(np.float32))
        ])
        # Debug summary
        try:
            s = out.select([
                pl.col(pred_col).mean().alias("mean"),
                pl.col(pred_col).std().alias("std"),
                pl.col(pred_col).n_unique().alias("unique"),
            ])
            print(f"DEBUG[APP] {pred_col} recentered (logit): mean={float(s[0,'mean']):.6f}, std={float(s[0,'std']):.6f}, unique={int(s[0,'unique'])}")
            # Show before/after comparison
            try:
                s0 = df.select([
                    pl.col(pred_col).mean().alias("mean"),
                    pl.col(pred_col).std().alias("std"),
                    pl.col(pred_col).n_unique().alias("unique"),
                ])
                print(f"DEBUG[APP] {pred_col} before:  mean={float(s0[0,'mean']):.6f}, std={float(s0[0,'std']):.6f}, unique={int(s0[0,'unique'])}")
            except Exception:
                pass
        except Exception:
            pass
        return out
    except Exception:
        return df

def validate_contract_predictions_against_training(df: pl.DataFrame, club_or_tournament: str) -> None:
    """
    Validate Contract predictions against saved training data (if available).
    This helps identify discrepancies between training and inference accuracy.
    """
    import pathlib
    import json
    
    # Check for debug files from training module (in the training module's location)
    acbl_path = pathlib.Path("e:/bridge/data/acbl")
    debug_input_file = acbl_path / "debug_input_contract.parquet"
    debug_predictions_file = acbl_path / "debug_predictions_contract.parquet"
    
    if not (debug_input_file.exists() and debug_predictions_file.exists()):
        print("🔍 Training debug files not found - skipping validation")
        return
    
    print("=" * 60)
    print("🔍 VALIDATING AGAINST TRAINING MODULE RESULTS")
    print("=" * 60)
    
    try:
        # Load training results
        training_input_df = pl.read_parquet(debug_input_file)
        training_predictions_df = pl.read_parquet(debug_predictions_file)
        
        # Calculate training accuracy
        if 'Contract' in training_predictions_df.columns and 'Contract_Pred' in training_predictions_df.columns:
            training_accuracy = (training_predictions_df['Contract'] == training_predictions_df['Contract_Pred']).mean()
            training_pred_counts = training_predictions_df['Contract_Pred'].value_counts().sort('Contract_Pred')
            
            print(f"📊 TRAINING MODULE RESULTS:")
            print(f"   Input shape: {training_input_df.shape}")
            print(f"   Predictions shape: {training_predictions_df.shape}")
            print(f"   Training accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
            print(f"   Unique predictions: {training_predictions_df['Contract_Pred'].n_unique()}")
            print(f"   Top predictions: {training_pred_counts.head(5).to_dict()}")
            
            # Re-run prediction on training input using current pipeline
            print(f"\n🔄 RE-RUNNING ON TRAINING INPUT...")
            model_name = f'acbl_{club_or_tournament}_predicted_contract_torch_model'
            
            # Add fake columns that predict_model expects
            # todo: put this fake stuff in an earlier step.
            test_df = training_input_df
            if 'MasterPoints_N' not in test_df.columns:
                print(f"Adding fake column MasterPoints_[NESW] to training input dataframe")
                test_df = test_df.with_columns([
        pl.lit(None).cast(pl.Float32).alias('MasterPoints_N'),
        pl.lit(None).cast(pl.Float32).alias('MasterPoints_E'),
        pl.lit(None).cast(pl.Float32).alias('MasterPoints_S'),
        pl.lit(None).cast(pl.Float32).alias('MasterPoints_W'),
                ])
            # Add missing numerical columns (these are all expected to be numerical according to schema)
            special_fixups = [('board_result_id',pl.Int64),('Club',pl.Int32),('Dealer',pl.Categorical),('Round',pl.UInt8),('Table',pl.UInt8),('section_id',pl.Int64),('tb_count',pl.Float32),('MP_Top',pl.Float32)]
            special_fixups += [('Pair_Number_EW',pl.UInt8),('Pair_Number_NS',pl.UInt8),('event_id',pl.Int64),('session_id',pl.Int64)] if club_or_tournament == 'club' else [('Pair_Number_EW',pl.Int32),('Pair_Number_NS',pl.Int32),('event_id',pl.String),('session_id',pl.String)]
            for col,dtype in special_fixups:
                if col not in test_df.columns:
                    print(f"Adding fake column {col} to training input dataframe")
                    # Use sensible defaults per dtype
                    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                        default_expr = pl.lit(0).cast(dtype)
                    elif dtype in (pl.Float32, pl.Float64):
                        default_expr = pl.lit(0.0).cast(dtype)
                    elif dtype == pl.Categorical:
                        default_expr = pl.lit(None).cast(pl.Categorical)
                    else:
                        default_expr = pl.lit(None).cast(dtype)
                    test_df = test_df.with_columns([ default_expr.alias(col) ])
                else:
                    if test_df[col].dtype != dtype:
                        print(f"Converting column {col} from {test_df[col].dtype} to {dtype}")
                        test_df = test_df.with_columns([
                            pl.col(col).cast(dtype).alias(col),
                        ])

            # Run current inference pipeline using training module's saved models
            training_saved_models_path = acbl_path / "SavedModels"
            new_prediction_df = mlBridgeAiLib.predict_model(training_saved_models_path, model_name, test_df)
            
            # Add actual Contract column for comparison
            new_prediction_df = new_prediction_df.with_columns([
                training_predictions_df['Contract'].alias('Contract')
            ])
            
            # Calculate current accuracy
            if 'Contract_Pred' in new_prediction_df.columns:
                current_accuracy = (new_prediction_df['Contract'] == new_prediction_df['Contract_Pred']).mean()
                current_pred_counts = new_prediction_df['Contract_Pred'].value_counts().sort('Contract_Pred')
                
                print(f"📊 CURRENT INFERENCE RESULTS:")
                print(f"   Current accuracy: {current_accuracy:.4f} ({current_accuracy*100:.2f}%)")
                print(f"   Unique predictions: {new_prediction_df['Contract_Pred'].n_unique()}")
                print(f"   Top predictions: {current_pred_counts.head(5).to_dict()}")
                
                # Compare results
                accuracy_diff = current_accuracy - training_accuracy
                print(f"\n📊 COMPARISON:")
                print(f"   Training accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
                print(f"   Current accuracy:  {current_accuracy:.4f} ({current_accuracy*100:.2f}%)")
                print(f"   Difference:        {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")
                
                if abs(accuracy_diff) > 0.05:  # More than 5% difference
                    print(f"   âŒ SIGNIFICANT ACCURACY DROP DETECTED!")
                    print(f"   This indicates a problem with the inference pipeline.")
                    
                    # Check schema for debugging
                    schema_path = training_saved_models_path / f"{model_name}_schema.json"
                    if schema_path.exists():
                        with open(schema_path, 'r') as f:
                            schema = json.load(f)
                        print(f"   Schema categorical features: {len(schema.get('categorical_feature_cols', []))}")
                        print(f"   Schema numerical features: {len(schema.get('numerical_feature_cols', []))}")
                        print(f"   Schema category mappings: {len(schema.get('category_mappings', {}))}")
                else:
                    print(f"   ✅ Accuracies match within tolerance.")
            
    except Exception as e:
        print(f"âŒ Error during validation: {str(e)}")
    
    print("=" * 60)

def _recenter_pct_ns_pred(df: pl.DataFrame, y_ns: str, label: str) -> pl.DataFrame:
    """Recenter Pct_NS_Pred to mean=0.5 with iterative adjustment for clipping effects."""
    pred_col = f'{y_ns}_Pred'
    current_mean = float(df[pred_col].mean())
    current_std = float(df[pred_col].std())
    n_rows = len(df)
    print_to_log_debug(f"DEBUG[APP] Recentering {label}: n_rows={n_rows}, BEFORE mean={current_mean:.6f}, std={current_std:.6f}")
    
    # Iteratively adjust shift to account for clipping effects
    target_mean = 0.5
    shift = target_mean - current_mean
    max_iterations = 10
    tolerance = 1e-6
    
    for iteration in range(max_iterations):
        # Apply shift and clip
        df_temp = df.with_columns([
            (pl.col(pred_col) + shift).clip(0.0, 1.0).alias(f'{pred_col}_temp')
        ])
        actual_mean = float(df_temp[f'{pred_col}_temp'].mean())
        
        # Check if we're close enough
        if abs(actual_mean - target_mean) < tolerance:
            df = df_temp.with_columns([
                pl.col(f'{pred_col}_temp').alias(pred_col)
            ]).drop(f'{pred_col}_temp')
            break
        
        # Adjust shift based on error
        shift += (target_mean - actual_mean)
    else:
        # Max iterations reached, use last result
        df = df_temp.with_columns([
            pl.col(f'{pred_col}_temp').alias(pred_col)
        ]).drop(f'{pred_col}_temp')
    
    after_mean = float(df[pred_col].mean())
    after_std = float(df[pred_col].std())
    print_to_log_debug(f"DEBUG[APP] Recentering {label}: AFTER mean={after_mean:.6f}, std={after_std:.6f}, shift={shift:.6f}")
    return df

def Predict_Game_Results(df: Any) -> Optional[Any]:
    if df is None:
        return None

    club_or_tournament = 'club' if 'club' in st.session_state.game_results_url else 'tournament'

    def to_pl_dtype(s: str):
        mapping = {
            'Float32': pl.Float32,
            'Float64': pl.Float64,
            'Int8': pl.Int8,
            'Int16': pl.Int16,
            'Int32': pl.Int32,
            'Int64': pl.Int64,
            'UInt8': pl.UInt8,
            'UInt16': pl.UInt16,
            'UInt32': pl.UInt32,
            'UInt64': pl.UInt64,
            'Boolean': pl.Boolean,
            'Categorical': pl.Categorical,
            'Utf8': pl.Utf8,
            'String': pl.Utf8,
            'Date': pl.Date,
            'Datetime': pl.Datetime,
        }
        if s not in mapping:
            raise ValueError(f"Unknown dtype '{s}' - add it to to_pl_dtype mapping")
        return mapping[s]

    assert df.select(pl.col('^Elo_.*$')).is_empty(), f"Oops. Found Elo columns in df: {df.select(pl.col('^Elo_.*$')).columns}"
    # takes 3s/0s for 13m/1m rows by 5 columns creating a 132MB/1MB? file.
    acbl_club_elo_ratings_filename = f'acbl_{club_or_tournament}_player_elo_ratings.parquet'
    acbl_club_elo_ratings_file = st.session_state.savedModelsPath.joinpath(acbl_club_elo_ratings_filename)
    player_elo_df = pl.read_parquet(acbl_club_elo_ratings_file)
    print(f"Loaded {acbl_club_elo_ratings_filename}: shape:{player_elo_df.shape} size:{acbl_club_elo_ratings_file.stat().st_size}")
    # takes 3s/0s for 13m/1m rows by 5 columns creating a 132MB/1MB? file.
    acbl_club_elo_ratings_filename = f'acbl_{club_or_tournament}_pair_elo_ratings.parquet'
    acbl_club_elo_ratings_file = st.session_state.savedModelsPath.joinpath(acbl_club_elo_ratings_filename)
    pair_elo_df = pl.read_parquet(acbl_club_elo_ratings_file)
    print(f"Loaded {acbl_club_elo_ratings_filename}: shape:{pair_elo_df.shape} size:{acbl_club_elo_ratings_file.stat().st_size}")

    # todo: put Elo code into mlBridgeAcblLib?
    # Deduplicate ELO dataframes to prevent row multiplication during joins
    # Keep the latest (highest Elo_N) rating when a player has multiple records for same date
    player_elo_df_dedup = (
        player_elo_df
        .sort(['Date', 'Player_ID', 'Elo_N'], descending=[False, False, True])
        .unique(subset=['Date', 'Player_ID'], keep='first', maintain_order=True)
    )
    # todo: change elo creation so there's a sorted Pair_IDs column already existing?
    # Convert list to sorted string for joining (join_asof doesn't support list columns)
    pair_elo_df = pair_elo_df.with_columns(
        pl.col('Pair_IDs').str.split('-').list.sort().list.join('-').alias('Pair_IDs_Sorted_Str')
    )
    pair_elo_df_dedup = (
        pair_elo_df
        .sort(['Date', 'Pair_IDs_Sorted_Str', 'Elo_N'], descending=[False, False, True])
        .unique(subset=['Date', 'Pair_IDs_Sorted_Str'], keep='first', maintain_order=True)
    )
    
    # Track initial row count to verify no duplicates
    initial_row_count = len(df)
    
    # Debug: Check data types before joining
    print(f"DEBUG: Main df dtypes - Date: {df['Date'].dtype}")
    print(f"DEBUG: Player ELO dtypes - Date: {player_elo_df_dedup['Date'].dtype}")
    print(f"DEBUG: Main df sample join keys:\n{df.select(['Date', 'Player_ID_N', 'Player_ID_E']).head(3)}")
    print(f"DEBUG: Player ELO sample join keys:\n{player_elo_df_dedup.select(['Date', 'Player_ID', 'Elo_N']).head(3)}")
    
    # Check if the specific date exists in ELO data
    main_date = df['Date'][0]
    print(f"\nDEBUG: Looking for Date={main_date}")
    matching_elo = player_elo_df_dedup.filter(pl.col('Date') == main_date)
    print(f"DEBUG: Found {len(matching_elo)} matching rows in player ELO for this date")
    if len(matching_elo) > 0:
        print(f"DEBUG: Sample matching ELO data:\n{matching_elo.head(5)}")
    
    # Check date range in ELO data
    print(f"DEBUG: Player ELO date range: {player_elo_df_dedup['Date'].min()} to {player_elo_df_dedup['Date'].max()}")
    
    # Join player ELO ratings for each direction
    # Note: We use join_asof to get the most recent ELO rating on or before the game date
    # This handles cases where the exact game date doesn't exist in the ELO data
    # ELO tracks a player's rating across all games, so we use the last known rating
    for direction in 'NESW':
        # Select only the exact columns we need to avoid conflicts
        elo_cols = player_elo_df_dedup.select(['Date','Player_ID','Elo_N','Elo_R_EventStart','MasterPoints']).rename({
            'Elo_R_EventStart': f'Elo_R_{direction}_EventStart',
            'Elo_N': f'Elo_N_{direction}',
            'MasterPoints': f'MasterPoints_{direction}',
        }).with_columns(pl.col('Date').cast(pl.Date)) # Ensure Date type matches for join
        
        # Use join_asof to get the most recent ELO rating on or before the game date
        df = df.join_asof(
            elo_cols, 
            left_on='Date',
            right_on='Date',
            by_left=f'Player_ID_{direction}',
            by_right='Player_ID',
            strategy='backward'  # Get the most recent date <= game date
        )
        assert df.select(pl.col(f'^.*_right$')).is_empty()
        print(f"DEBUG: After joining {direction}, null count for Elo_N_{direction}: {df[f'Elo_N_{direction}'].null_count()}")
    # Verify no row multiplication
    if len(df) != initial_row_count:
        print(f"WARNING: Row count changed from {initial_row_count} to {len(df)} after player ELO joins!")
    
    # Join pair ELO ratings for NS and EW
    # Note: We use join_asof to get the most recent ELO rating on or before the game date
    # This handles cases where the exact game date doesn't exist in the ELO data
    for pair in ['NS', 'EW']:
        # Select only the exact columns we need to avoid conflicts
        elo_cols = pair_elo_df_dedup.select(['Date','Pair_IDs_Sorted_Str','Elo_N','Elo_R_EventStart']).rename({
            'Elo_R_EventStart': f'Elo_R_{pair}_EventStart',
            'Elo_N': f'Elo_N_{pair}',
        }).with_columns(pl.col('Date').cast(pl.Date))
        
        # Create sorted string version of Pair_IDs for joining
        df = df.with_columns(
            pl.col(f'Pair_IDs_{pair}').list.sort().list.join('-').alias(f'Pair_IDs_{pair}_Sorted_Str')
        )
        
        # Use join_asof to get the most recent ELO rating on or before the game date
        df = df.join_asof(
            elo_cols, 
            left_on='Date',
            right_on='Date',
            by_left=f'Pair_IDs_{pair}_Sorted_Str',
            by_right='Pair_IDs_Sorted_Str',
            strategy='backward'  # Get the most recent date <= game date
        )
        
        # Drop the temporary string column used for joining
        df = df.drop(f'Pair_IDs_{pair}_Sorted_Str')
        
        assert df.select(pl.col(f'^.*_right$')).is_empty()
        print(f"DEBUG: After joining {pair}, null count for Elo_N_{pair}: {df[f'Elo_N_{pair}'].null_count()}")

    # Final verification
    if len(df) != initial_row_count:
        print(f"WARNING: Final row count is {len(df)}, expected {initial_row_count}!")

    for y_name in ['Declarer_Direction', 'Contract', 'Pct_NS']:
        model_name = f'acbl_{club_or_tournament}_predicted_{y_name.lower()}_torch_model'
        with st.spinner(f"Predicting {y_name}..."):
            schema_path = st.session_state.savedModelsPath.joinpath(f"{model_name}_schema.json")
            if not schema_path.exists():
                st.error(f"Skipping {model_name} because {schema_path} not found.")
                continue
            schema = json.load(open(schema_path, 'r'))
            feature_dtypes = schema.get('feature_dtypes', {})
            required_features = schema.get('numerical_feature_cols', []) + schema.get('categorical_feature_cols', [])
            
            # First pass: Create None columns for any missing features
            for col in required_features:
                if col not in df.columns:
                    expected_dtype = to_pl_dtype(str(feature_dtypes[col]))
                    print(f"Adding fake column {col} of None with dtype {expected_dtype}")
                    df = df.with_columns([pl.lit(None).cast(expected_dtype).alias(col)])
            
            # Second pass: Ensure all columns have correct dtypes
            for col in required_features:
                expected_dtype = to_pl_dtype(str(feature_dtypes[col]))
                if df[col].dtype != expected_dtype:
                    print(f"Converting column {col} from {df[col].dtype} to {expected_dtype}")
                    df = df.with_columns([pl.col(col).cast(expected_dtype, strict=False).alias(col)])
                if df[col].is_null().all():
                    print(f"Column {col} is all nulls.")
            # Predict using library (schema determines model type)
            pred_df = mlBridgeAiLib.predict_model(
                st.session_state.savedModelsPath,
                model_name,
                df,
                top_k=5 if y_name == 'Contract' else 1,
                return_probs=(y_name == 'Contract')
            )

            # Debug prediction head/tail and stats for Pct_NS
            try:
                    stats_before = pred_df.select([
                        pl.col(f'{y_name}_Pred').mean().alias('mean'),
                        pl.col(f'{y_name}_Pred').std().alias('std'),
                        pl.col(f'{y_name}_Pred').min().alias('min'),
                        pl.col(f'{y_name}_Pred').max().alias('max'),
                        pl.col(f'{y_name}_Pred').n_unique().alias('unique'),
                    ])
                    print(f"DEBUG[APP] pred_df {y_name}_Pred stats: mean={float(stats_before[0,'mean']):.6f}, std={float(stats_before[0,'std']):.6f}, min={float(stats_before[0,'min']):.6f}, max={float(stats_before[0,'max']):.6f}, unique={int(stats_before[0,'unique'])}")
                    print(f"DEBUG[APP] pred_df {y_name}_Pred head: {pred_df.select(pl.col(f'{y_name}_Pred')).head(5).to_dict(as_series=False)}")
                    print(f"DEBUG[APP] pred_df {y_name}_Pred tail: {pred_df.select(pl.col(f'{y_name}_Pred')).tail(5).to_dict(as_series=False)}")
            except Exception:
                pass

            # Merge predictions
            overlapping = [c for c in pred_df.columns if c in df.columns]
            if overlapping:
                df = df.drop(overlapping)
            df = df.hstack(pred_df)

            # Minimal post-processing per target
            if y_name == 'Pct_NS' and f'{y_name}_Pred' in df.columns:
                y_ns = y_name
                y_ew = y_ns.replace('NS', 'EW')
                df = df.with_columns([
                    pl.col(y_ns).alias(f'{y_ns}_Actual'),
                    (1 - pl.col(y_ns)).alias(y_ew),
                    (1 - pl.col(y_ns)).alias(f'{y_ew}_Actual'),  # Fixed: use original Pct_NS, not derived y_ew
                    (1 - pl.col(f'{y_ns}_Pred')).alias(f'{y_ew}_Pred'),
                ])
                # Linear recentering to mean=0.5 PER GAME/SECTION (with iterative adjustment for clipping effects)
                # Must recenter within each game to preserve game-level mean=0.5
                # Try multiple grouping columns: section_id, event_id, session_id
                session_col = None
                for col in ['section_id', 'Section_ID', 'event_id', 'Event_ID', 'session_id', 'Session_ID']:
                    if col in df.columns:
                        session_col = col
                        break
                
                if session_col and df[session_col].n_unique() > 1:
                    # Multiple sessions: recenter each separately
                    print_to_log_debug(f"DEBUG[APP] Recentering {df[session_col].n_unique()} sessions separately")
                    recentered_dfs = []
                    for session_id in df[session_col].unique().sort():
                        session_df = df.filter(pl.col(session_col) == session_id)
                        session_df = _recenter_pct_ns_pred(session_df, y_ns, f"session {session_id}")
                        recentered_dfs.append(session_df)
                    df = pl.concat(recentered_dfs)
                else:
                    # Single session or no session column: recenter all together
                    df = _recenter_pct_ns_pred(df, y_ns, "all data")
                
                # Update EW predictions after recentering (must be done for both branches)
                df = df.with_columns([
                    (1.0 - pl.col(f'{y_ns}_Pred')).alias(f'{y_ew}_Pred')
                ])
                
                after_mean = float(df[f'{y_ns}_Pred'].mean())
                after_std = float(df[f'{y_ns}_Pred'].std())
                after_sum = float(df[f'{y_ns}_Pred'].sum())
                n_rows = len(df)
                print_to_log_debug(f"DEBUG[APP] Pct_NS_Pred AFTER recenter (overall): n_rows={n_rows}, mean={after_mean:.6f}, std={after_std:.6f}, sum={after_sum:.2f}, min={float(df[f'{y_ns}_Pred'].min()):.6f}, max={float(df[f'{y_ns}_Pred'].max()):.6f}, unique={df[f'{y_ns}_Pred'].n_unique()}")
                
                # Debug: Check per-pair averages to see if they preserve mean=0.5
                pair_col = None
                for col in ['Pair_NS', 'pair_ns', 'NS_Pair', 'ns_pair', 'Player_Pair_NS', 'player_pair_ns']:
                    if col in df.columns:
                        pair_col = col
                        print_to_log_debug(f"DEBUG[APP] Found pair column: {pair_col}")
                        break
                
                if pair_col:
                    try:
                        pair_stats = df.group_by(pair_col).agg([
                            pl.col(f'{y_ns}_Pred').mean().alias('Avg_Pct_NS_Pred'),
                            pl.col(f'{y_ns}_Pred').count().alias('Board_Count')
                        ]).sort('Avg_Pct_NS_Pred', descending=True)
                        
                        # Simple mean of pair averages (unweighted)
                        pair_mean_unweighted = float(pair_stats['Avg_Pct_NS_Pred'].mean())
                        pair_std = float(pair_stats['Avg_Pct_NS_Pred'].std())
                        pair_min = float(pair_stats['Avg_Pct_NS_Pred'].min())
                        pair_max = float(pair_stats['Avg_Pct_NS_Pred'].max())
                        
                        # Weighted mean of pair averages (weighted by board count)
                        total_predictions = pair_stats['Board_Count'].sum()
                        pair_mean_weighted = float((pair_stats['Avg_Pct_NS_Pred'] * pair_stats['Board_Count']).sum() / total_predictions)
                        
                        print_to_log_debug(f"DEBUG[APP] Per-pair NS averages: n_pairs={len(pair_stats)}, unweighted_mean={pair_mean_unweighted:.6f}, weighted_mean={pair_mean_weighted:.6f}, std={pair_std:.6f}, range=[{pair_min:.3f}, {pair_max:.3f}]")
                        print_to_log_debug(f"DEBUG[APP] Board counts per pair: min={pair_stats['Board_Count'].min()}, max={pair_stats['Board_Count'].max()}, total={total_predictions}")
                    except Exception as e:
                        print_to_log_debug(f"DEBUG[APP] Error computing pair stats: {e}")
                else:
                    # List all columns that contain 'pair' or 'ns' (case insensitive)
                    pair_like_cols = [c for c in df.columns if 'pair' in c.lower() or ('ns' in c.lower() and 'pct' not in c.lower())]
                    print_to_log_debug(f"DEBUG[APP] No standard pair column found. Pair-like columns: {pair_like_cols[:10]}")
                # Sanity-check after recentering
                try:
                    s_after = df.select([
                        pl.col(f'{y_ns}_Pred').mean().alias('mean'),
                        pl.col(f'{y_ns}_Pred').std().alias('std'),
                        pl.col(f'{y_ns}_Pred').n_unique().alias('unique'),
                    ])
                    print(f"DEBUG[APP] {y_ns}_Pred after recenter: mean={float(s_after[0,'mean']):.6f}, std={float(s_after[0,'std']):.6f}, unique={int(s_after[0,'unique'])}")
                except Exception:
                    pass
                df = df.with_columns([
                    (pl.col(y_ns) - pl.col(f'{y_ns}_Pred')).alias(f'{y_ns}_Pred_Error'),
                    (pl.col(y_ns) - pl.col(f'{y_ns}_Pred')).abs().alias(f'{y_ns}_Pred_Absolute_Error'),
                    (pl.col(y_ew) - pl.col(f'{y_ew}_Pred')).alias(f'{y_ew}_Pred_Error'),
                    (pl.col(y_ew) - pl.col(f'{y_ew}_Pred')).abs().alias(f'{y_ew}_Pred_Absolute_Error'),
                ])
            elif y_name == 'Declarer_Direction' and f'{y_name}_Pred' in df.columns:
                df = df.with_columns([
                    pl.struct([f'{y_name}_Pred', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W'])
                      .map_elements(lambda row: row[f'Player_ID_{row[f"{y_name}_Pred"]}'] if row[f"{y_name}_Pred"] in 'NESW' else None, return_dtype=pl.String)
                      .alias('Declarer_ID_Pred'),
                    pl.struct([f'{y_name}_Pred', 'Player_Name_N', 'Player_Name_E', 'Player_Name_S', 'Player_Name_W'])
                      .map_elements(lambda row: row[f'Player_Name_{row[f"{y_name}_Pred"]}'] if row[f"{y_name}_Pred"] in 'NESW' else None, return_dtype=pl.String)
                      .alias('Declarer_Name_Pred'),
                ])

    return df

def ensure_board_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure board-scoped boolean flags exist (e.g., Boards_We_Played, Boards_I_Declared).
    Reconstructs from session context if missing.
    """
    try:
        all_flags = ['Boards_We_Played', 'Boards_I_Played', 'Our_Section', 'Our_Pair', 'My_Section', 'My_Pair',
                     'Boards_I_Declared', 'Boards_Partner_Declared', 'Boards_We_Declared', 'Boards_Opponent_Declared', 'Our_Boards']
        missing_flags = [c for c in all_flags if c not in df.columns]
        if not missing_flags:
            return df
        # Build My_Section if missing
        if 'My_Section' not in df.columns:
            df = df.with_columns([
                (pl.col('section_name') == pl.lit(st.session_state.section_name)).alias('My_Section'),
            ])
        # Build My_Pair if missing
        if 'My_Pair' not in df.columns and st.session_state.get('pair_direction') and st.session_state.get('pair_number') is not None:
            pair_col = f"Pair_Number_{st.session_state.pair_direction}"
            if pair_col in df.columns:
                df = df.with_columns([
                    (pl.col('My_Section') & (pl.col(pair_col) == pl.lit(st.session_state.pair_number))).alias('My_Pair'),
                ])
        # Derive additional flags from My_Pair
        if 'Our_Section' not in df.columns:
            df = df.with_columns(pl.col('My_Section').alias('Our_Section'))
        if 'Our_Pair' not in df.columns:
            df = df.with_columns(pl.col('My_Pair').alias('Our_Pair'))
        if 'Our_Boards' not in df.columns:
            df = df.with_columns(pl.col('My_Pair').alias('Our_Boards'))
        if 'Boards_I_Played' not in df.columns:
            df = df.with_columns(pl.col('My_Pair').alias('Boards_I_Played'))
        if 'Boards_We_Played' not in df.columns:
            df = df.with_columns(pl.col('My_Pair').alias('Boards_We_Played'))
        
        # Build declarer-specific flags (requires Declarer_ID column)
        if 'Declarer_ID' in df.columns:
            if 'Boards_I_Declared' not in df.columns and st.session_state.get('player_id'):
                df = df.with_columns(
                    (pl.col('My_Pair') & (pl.col('Declarer_ID') == st.session_state.player_id)).alias('Boards_I_Declared')
                )
            if 'Boards_Partner_Declared' not in df.columns and st.session_state.get('partner_id'):
                df = df.with_columns(
                    (pl.col('My_Pair') & (pl.col('Declarer_ID') == st.session_state.partner_id)).alias('Boards_Partner_Declared')
                )
            if 'Boards_Opponent_Declared' not in df.columns and st.session_state.get('opponent_pair_direction'):
                opponent_pair_direction = st.session_state.opponent_pair_direction
                df = df.with_columns(
                    (pl.col('My_Pair') & ((pl.col('Declarer_Direction') == opponent_pair_direction[0]) | 
                     (pl.col('Declarer_Direction') == opponent_pair_direction[1]))).alias('Boards_Opponent_Declared')
                )
            if 'Boards_We_Declared' not in df.columns and 'Boards_I_Declared' in df.columns and 'Boards_Partner_Declared' in df.columns:
                df = df.with_columns(
                    (pl.col('Boards_I_Declared') | pl.col('Boards_Partner_Declared')).alias('Boards_We_Declared')
                )
        return df
    except Exception as e:
        print(f"Error in ensure_board_flags: {e}")
        return df


def run_ai_predictions_now() -> None:
    """Run AI predictions on the currently loaded dataframe and re-register it, or load from cache."""

    # Check if predictions are already cached
    if st.session_state.get('predictions_cached', False) and st.session_state.get('predicted_df') is not None:
        st.info("AI predictions already cached. Displaying previous results.")
        st.session_state.df = st.session_state.predicted_df  # Load cached predictions back into df
        # IMPORTANT: Call ensure_board_flags() on cached data too!
        st.session_state.df = ensure_board_flags(st.session_state.df)
        # Don't return yet - need to re-register with DuckDB!
    else:
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("Load a game first before running AI predictions.")
            return
        df = st.session_state.df
        with st.spinner('Making AI Predictions. Takes 30 seconds.'):
            t = time.time()
            df_with_predictions = Predict_Game_Results(df)
            if df_with_predictions is None:
                st.error("AI predictions failed. Cannot continue without predictions.")
                st.stop()
            # Cache the predictions
            st.session_state.predicted_df = df_with_predictions
            st.session_state.predictions_cached = True

            # Ensure flag columns exist before registering
            st.session_state.df = ensure_board_flags(df_with_predictions)
            print_to_log_info('AI Predictions (on-demand) time:', time.time()-t)
    
    # ALWAYS re-register DuckDB table (whether cached or freshly computed)
    con = get_db_connection()
    table_name = st.session_state.con_register_name
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
    except Exception:
        pass
    assert st.session_state.df.select(pl.col(pl.Object)).is_empty(), f"Found Object columns: {st.session_state.df.select(pl.col(pl.Object)).columns}"
    st.session_state.df = cast_numeric_display_columns(st.session_state.df)
    con.register(table_name, st.session_state.df)
    # Update schema string
    st.session_state.df_schema_string = create_schema_string(st.session_state.df, con)
    with open('df_schema.sql','w') as f:
        f.write(st.session_state.df_schema_string)
    
    # Reset the flag so predictions don't run again when switching games/sessions
    st.session_state.ai_predictions_requested = False


# def reset_data():
#     # resets all data. used initially and when player number changes.
#     # todo: put all session state into st.session_state.data so that clearing data clears all session states.

#     print_to_log_info('reset_data()')

#     # app
#     #st.session_state.app_datetime = None
#     st.session_state.help = None
#     st.session_state.release_notes = None

#     # game
#     st.session_state.game_date = None
#     st.session_state.session_id = None
#     st.session_state.acbl_results_page = None
#     st.session_state.json_results_d = None

#     # chat
#     #st.session_state.ai_api = None
#     #st.session_state.system_prompt = None
#     #st.session_state.augmented_system_prompt = None
#     #st.session_state.messages = []
#     #st.session_state.dataframes = defaultdict(list)
#     #st.session_state.matchpoint_ns_d = None
#     #st.session_state.function_calls = None

#     # data
#     st.session_state.df = None
#     st.session_state.do_not_cache_df = True

#     # sql
#     #st.session_state.con = None
#     st.session_state.con_register_name = 'self' # todo: this and others have duplicate initializations.
#     #st.session_state.show_sql_query = None
#     #st.session_state.commands_sql = None
#     st.session_state.df_meta = None
#     st.session_state.df_schema_string = None

#     # favorite files
#     st.session_state.favorites = None
#     st.session_state.default_favorites_file = None
#     st.session_state.player_id_favorites = None
#     st.session_state.player_id_custom_favorites_file = None
#     st.session_state.debug_favorites = None
#     st.session_state.debug_favorites_file = None
#     st.session_state.prompts_selectbox = 'Choose a Prompt'
#     #st.session_state.vetted_prompts = None
#     st.session_state.vetted_prompt_titles = None
#     st.session_state.dataframe_tooltips = None

#     # augmented columns
#     st.session_state.player_id = None
#     st.session_state.player_direction = None
#     st.session_state.player_name = None
#     st.session_state.player_id = None
#     st.session_state.partner_direction = None
#     st.session_state.partner_name = None
#     st.session_state.pair_number = None
#     st.session_state.pair_direction = None
#     st.session_state.pair_name = None
#     st.session_state.opponent_pair_direction = None
#     st.session_state.session_id = None
#     st.session_state.section_name = None

    # main_message_df_count = 0
    # for k, v in st.session_state.items():
    #     print_to_log_info('session_state:',k)
    #     if k.startswith('main_messages_df_'):
    #         # assert st.session_state[k] is None # This happened once on 29-Sep-2023. Not sure why. Maybe there's a timing issue with st.session_state and st.container being destroyed?
    #         #del st.session_state[k] # delete the key. This is a hack. It's not clear why the key is not being deleted when the container is destroyed.
    #         main_message_df_count += 1
    # print_to_log_info('main_message_df_: count:',main_message_df_count,st.session_state.df_unique_id)

    # # These files are repeatedly reloaded for development purposes. Only takes a second.

    # system_prompt_file = pathlib.Path('system_prompt.txt')
    # if not system_prompt_file.exists():
    #     st.write(f"Oops. {system_prompt_file} file does not exist.")
    #     st.stop()
    # with open(system_prompt_file, 'r') as f:
    #     system_prompt = f.read()  # text string
    #     st.session_state.system_prompt = system_prompt

    # commands_sql_file = pathlib.Path('commands.sql')
    # if commands_sql_file.exists():
    #     with open(commands_sql_file, 'r') as f:
    #         commands_sql = f.read()  # text string
    #         st.session_state.commands_sql = commands_sql

    # todo: feature removed
    # ai_apis_file = pathlib.Path('ai_apis.json')
    # if ai_apis_file.exists():
    #     with open(ai_apis_file, 'r') as f:
    #         ai_apis = json.load(f)  # dict
    #         st.session_state.ai_apis = ai_apis['AI_APIs']['Models']
    #         assert len(st.session_state.ai_apis) > 0 and DEFAULT_AI_MODEL in st.session_state.ai_apis, f"Oops. {DEFAULT_AI_MODEL} not in {st.session_state.ai_apis}."
    # else:
    #     st.session_state.ai_apis = [DEFAULT_AI_MODEL]
    # st.session_state.ai_api = DEFAULT_AI_MODEL


def create_sidebar() -> None:
    
    t = time.time()

    st.sidebar.caption(f"Build:{st.session_state.app_datetime}")

    # Check if we need to handle validation failure
    validation_failed = st.session_state.get('player_id_validation_failed', False)
    
    # Determine what value to show in the text input
    if validation_failed:
        # Show the previous valid player ID when validation fails
        input_value = st.session_state.get('player_id', '')
    else:
        # Normal case - use whatever is currently in the input
        input_value = st.session_state.get('player_id_input', st.session_state.get('player_id', ''))

    st.sidebar.text_input(
        "ACBL player number", 
        value=input_value,
        on_change=player_id_change, 
        placeholder=st.session_state.player_id_default, 
        key='player_id_input')

    # Handle player ID validation failure (after widget creation)
    if validation_failed:
        # Clear any existing report from the main window
        if hasattr(st.session_state, 'main_section_container'):
            st.session_state.main_section_container = st.empty()
            # Create a new container with a helpful message
            st.session_state.main_section_container = st.container()
            with st.session_state.main_section_container:
                st.info("Invalid player ID entered. Please enter a valid ACBL player number in the sidebar to generate a new report.")
        # Clear SQL query mode and queries to prevent confusion
        st.session_state.sql_query_mode = False
        st.session_state.sql_queries = []
        # Clear player-specific session data to prevent confusion
        st.session_state.session_id = None
        st.session_state.df = None
        if hasattr(st.session_state, 'game_description'):
            st.session_state.game_description = None
        if hasattr(st.session_state, 'player_name'):
            st.session_state.player_name = None
        if hasattr(st.session_state, 'partner_name'):
            st.session_state.partner_name = None
        # Show additional error context if available
        if 'invalid_player_id_input' in st.session_state:
            st.sidebar.error(f"Invalid player ID: {st.session_state.invalid_player_id_input}")
            del st.session_state.invalid_player_id_input
        # Clear the validation failure flag after handling
        st.session_state.player_id_validation_failed = False

    # Load configuration files early so Developer Settings has access to debug_favorites
    # These files are reloaded each time for development purposes. Only takes a second.
    st.session_state.default_favorites_file = pathlib.Path('default.favorites.json')
    if st.session_state.player_id is not None:
        st.session_state.player_id_custom_favorites_file = pathlib.Path(
            'favorites/'+st.session_state.player_id+'.favorites.json')
    st.session_state.debug_favorites_file = pathlib.Path('favorites/debug.favorites.json')
    read_configs()

    if st.session_state.player_id is None:
        # Show message and then fall through to Developer Settings at bottom
        st.sidebar.info("Enter a player ID above to view game reports.")
        # Don't return early - let Developer Settings and Automated Postmortem Apps show at bottom
    else:
        # Player ID is set - show game selection and other player-specific UI.
        # Compute selectbox indices from the currently active session_id so a
        # URL-supplied or callback-set session_id is reflected in the dropdowns.
        player_games = st.session_state.game_urls_d.get(st.session_state.player_id, {})
        player_tournaments = st.session_state.tournament_session_urls_d.get(st.session_state.player_id, {})
        current_sid = st.session_state.get('session_id')

        club_index = 0 if player_games else None
        tournament_index = None
        if current_sid is not None:
            sid_str = str(current_sid)
            for i, key in enumerate(player_games.keys()):
                if str(key) == sid_str:
                    club_index = i
                    break
            for i, key in enumerate(player_tournaments.keys()):
                if str(key) == sid_str:
                    tournament_index = i
                    break

        st.sidebar.selectbox(
            "Choose a club game.",
            index=club_index,
            options=[f"{k}, {v[2]}" for k, v in player_games.items()],
            on_change=club_session_id_change,
            key='club_session_ids_selectbox',
        )  # options are event_id + event description

        st.sidebar.selectbox(
            "Choose a tournament session.",
            index=tournament_index,
            options=[f"{k}, {v[2]}" for k, v in player_tournaments.items()],
            on_change=tournament_session_id_change,
            key='tournament_session_ids_selectbox',
        )  # options are event_id + event description

        # Check if player ID just changed and we should auto-generate report for the first game
        if st.session_state.get('player_id_just_changed', False):
            st.session_state.player_id_just_changed = False
            # Clear the flag and generate the report automatically
            if st.session_state.session_id is not None:
                # Report will be generated in create_ui()
                pass
        
        if st.session_state.session_id is None:
            st.error(f'Please choose a new game or tournament session from the left sidebar.')
            # Don't return - fall through to Developer Settings at bottom
        else:
            # Session is selected - show session-specific UI
            # if st.sidebar.download_button(label="Download Personalized Report",
            #         data=streamlitlib.create_pdf(st.session_state.pdf_assets, title=f"Bridge Game Postmortem Report Personalized for {st.session_state.player_id}"),
            #         file_name = f"{st.session_state.session_id}-{st.session_state.player_id}-morty.pdf",
            #        mime='application/octet-stream',
            #         key='personalized_report_download_button'):
            #     st.warning('Personalized report downloaded.')

            help_file = pathlib.Path('help.md')
            if help_file.exists():
                with open(help_file, 'r') as f:
                    st.session_state.help = f.read()  # text string

            release_notes_file = pathlib.Path('release_notes.md')
            if release_notes_file.exists():
                with open(release_notes_file, 'r') as f:
                    st.session_state.release_notes = f.read()  # text string

            # Create placeholder for PDF link (will be populated after report is generated)
            st.session_state.pdf_link = st.sidebar.empty()
            
            # Show ACBL results page link
            if st.session_state.session_id is not None:
                if st.session_state.session_id in st.session_state.game_urls_d[st.session_state.player_id]:
                    st.session_state.acbl_results_page = st.session_state.game_urls_d[st.session_state.player_id][st.session_state.session_id][1]
                    markdown_acbl_results_page = f"[ACBL Club Result Page]({st.session_state.acbl_results_page})"
                    st.sidebar.markdown(markdown_acbl_results_page, unsafe_allow_html=True)
                elif st.session_state.session_id in st.session_state.tournament_session_urls_d[st.session_state.player_id]:
                    st.session_state.acbl_results_page = st.session_state.tournament_session_urls_d[st.session_state.player_id][st.session_state.session_id][1]
                    markdown_acbl_results_page = f"[ACBL Tournament Result Page]({st.session_state.acbl_results_page})"
                    st.sidebar.markdown(markdown_acbl_results_page, unsafe_allow_html=True)

            st.sidebar.divider()

            st.sidebar.write(
                'Below are favorite prompts. Either click a button below or enter a question in the prompt box at the bottom of the main section to the right.')

            if st.session_state.favorites is not None:
                st.sidebar.write('Favorite Prompts')

                # favorite buttons
                selected_boxes_vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']
                st.session_state.vetted_prompt_titles = {prompt['title']:prompt for prompt in st.session_state.favorites['SelectBoxes']['Vetted_Prompts'].values()}
                #st.session_state.vetted_prompts = {}
                for k, button in st.session_state.favorites['Buttons'].items():
                    # create dict of vetted prompts
                    # default list of vetted prompts is
                    if k == st.session_state.button_title_default:
                        st.session_state.selected_button = button
                    if st.sidebar.button(button['title'], help=button['help'], key=k):
                        st.session_state.sql_query_mode = False
                        st.session_state.selected_button = button
                        # If the button is AI Predictions, set request flag and run predictions now
                        if button['title'].lower().strip() == 'ai predictions':
                            st.session_state.ai_predictions_requested = True
                            run_ai_predictions_now()
                        
                        # 4 is arbitrary. clearning conversation so it doesn't become overwhelming to user or ai.
                        # if len(ups) > 4:
                        #     reset_messages()
                        #     ask_questions_without_context(
                        #         ups, st.session_state.ai_api)
                        #     #st.rerun() # this caused some systems to loop. not sure why.
                        # else:
                        #     ask_questions_without_context(ups, st.session_state.ai_api)
                
                st.session_state.dataframe_tooltips = {
                    col: tip for col, tip in st.session_state.favorites['ToolTips'].items()
                }

            # todo: reimplement user-specific favorites
            # if st.session_state.player_id_favorites is not None:
            #     st.sidebar.write(
            #         f"Player Number {st.session_state.player_id} Favorites")

            #     # player number favorite buttons
            #     for k, button in st.session_state.player_id_favorites['Buttons'].items():
            #         if st.sidebar.button(button['title'], help=button['help'], key=k):
            #             # temp - re-read for every button click for realtime debugging.
            #             #read_favorites()
            #             ups = []
            #             for up in st.session_state.player_id_favorites['Buttons'][k]['prompts']:
            #                 if up.startswith('@'):
            #                     box = st.session_state.vetted_prompts[up[1:]]
            #                     ups.append(box['prompts']) # create list of lists in case prompts are dependent on previous prompts
            #                 else:
            #                     ups.append(up)
            #             # ask_questions_without_context(ups, st.session_state.ai_api)

    # Automated Postmortem Apps - Always show (above Developer Settings at bottom of sidebar)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Automated Postmortem Apps**")
    st.sidebar.markdown("🔗 [ACBL Postmortem](https://acbl.postmortem.chat)")
    st.sidebar.markdown("🔗 [French ffbridge Postmortem](https://ffbridge.postmortem.chat)")
    st.sidebar.markdown("🔗 [Calculate PBN](https://pbn.postmortem.chat)")

    # Developer Settings - Always show regardless of player_id (at very bottom of sidebar)
    with st.sidebar.expander('Developer Settings', False):
        if st.session_state.debug_favorites is not None:
            # favorite prompts selectboxes
            st.session_state.debug_player_id_names = st.session_state.debug_favorites[
                'SelectBoxes']['Player_IDs']['options'] # todo: rename to Pair_IDs?
            if len(st.session_state.debug_player_id_names):
                # changed placeholder to player_id because when selectbox gets reset, possibly due to expander auto-collapsing, we don't want an unexpected value.
                # test player_id is not None else use debug_favorites['SelectBoxes']['player_ids']['placeholder']?
                st.selectbox("Debug Player List", options=st.session_state.debug_player_id_names, placeholder=st.session_state.player_id, #.debug_favorites['SelectBoxes']['player_ids']['placeholder'],
                                        on_change=debug_player_id_names_change, key='debug_player_id_names_selectbox')

                # Handle debug player ID validation failure
                if st.session_state.get('debug_player_id_validation_failed', False):
                    # Clear the validation failure flag
                    st.session_state.debug_player_id_validation_failed = False
                    # Clear any existing report from the main window
                    if hasattr(st.session_state, 'main_section_container'):
                        st.session_state.main_section_container = st.empty()
                        # Create a new container with a helpful message
                        st.session_state.main_section_container = st.container()
                        with st.session_state.main_section_container:
                            st.info("Invalid debug player ID selected. Please select a valid player from the debug list to generate a new report.")
                    # Clear SQL query mode and queries to prevent confusion
                    st.session_state.sql_query_mode = False
                    st.session_state.sql_queries = []
                    # Clear player-specific session data to prevent confusion
                    st.session_state.session_id = None
                    st.session_state.df = None
                    if hasattr(st.session_state, 'game_description'):
                        st.session_state.game_description = None
                    if hasattr(st.session_state, 'player_name'):
                        st.session_state.player_name = None
                    if hasattr(st.session_state, 'partner_name'):
                        st.session_state.partner_name = None
                    st.sidebar.error("Invalid debug player ID selected")
                    # Clear the validation failure flag after handling
                    st.session_state.debug_player_id_validation_failed = False

        st.checkbox(
            "Show SQL Queries", value=st.session_state.show_sql_query, on_change=show_sql_query_change, key='sql_query_checkbox')

        # Not at all fast to calculate. approximately .25 seconds per unique pbn overhead is minimum + .05 seconds per observation per unique pbn. e.g. time for 24 boards = 24 * (.25 + num of observations * .05).
        st.number_input(
            "Single Dummy Samples Count",
            min_value=1,
            max_value=100,
            value=st.session_state.single_dummy_sample_count,
            on_change=single_dummy_sample_count_changed,
            key='single_dummy_sample_count_number_input'
        )

    print_to_log_info('create_sidebar time:', time.time()-t)
    return

# def create_tab_bar():

#     t = time.time()
#     with st.container():

#         chat_tab, data, dtypes, schema, commands_sql, URLs, system_prompt_tab, favorites, help, release_notes, about, debug = st.tabs(
#             ['Chat', 'Data', 'dtypes', 'Schema', 'SQL', 'URLs', 'Sys Prompt', 'Favorites', 'Help', 'Release Notes', 'About', 'Debug'])
#         streamlitlib.stick_it_good()

#         with chat_tab:
#             pass

#         with data:
#             pass
#             #if st.session_state.df is not None:
#                 # AgGrid unreliable in displaying within tab so using st.dataframe instead
#                 # todo: why? Neil's event 846812 causes id error. must be NaN? # .style.format({col:'{:,.2f}' for col in st.session_state.df.select_dtypes('float')}))
#             #   ShowDataFrameTable(st.session_state.df, key='data_tab_df', tooltips=st.session_state.dataframe_tooltips)
#                 #st.dataframe(st.session_state.df)

#         with dtypes:
#             pass
#             # AgGrid unreliable in displaying within tab. Also issue with Series.
#             # gave 'Serialization of dataframe to Arrow table was unsuccessful' .astype('string') was appended.
#             #st.dataframe(st.session_state.df.to_pandas().dtypes.astype('string'))

#         with schema:
#             if st.session_state.df_schema_string is not None:
#                 # st.dataframe(st.session_state.df_meta)
#                 st.write(st.session_state.df_schema_string)
#                 # todo: index column shows twice. once as index and once as column. fix.
#                 # st.divider()
#                 # st.dataframe(pd.concat(
#                 #    [st.session_state.df.columns.to_series(), st.session_state.df.dtypes.name], axis='columns')) # gave arrow conversion error until .name was appended.

#         with commands_sql:
#             st.header('SQL Commands')
#             #st.write('SQL commands are not yet editable. Use the SQL commands to explore the data.')
#             st.write(st.session_state.commands_sql)

#         with URLs:
#             st.write(
#                 f"Player number is {st.session_state.player_id}")
#             st.divider()
#             st.write('Club Game URLs')
#             st.write(st.session_state.game_urls_d[st.session_state.player_id].values())
#             st.write('Tournament Sessions')
#             st.write(st.session_state.tournament_session_urls_d[st.session_state.player_id].values())

#         with system_prompt_tab:
#             st.header('System Prompt')
#             # todo: make system prompt editable. useful for experimenting.
#             #st.write('The system prompt is not yet editable.')
#             st.write(st.session_state.augmented_system_prompt)

#         with favorites:
#             read_favorites()  # todo: update each time for debugging
#             st.header(
#                 f"Default Favorites:{st.session_state.default_favorites_file}")
#             if st.session_state.favorites is not None:
#                 st.write(st.session_state.favorites)
#             st.divider()
#             st.header(
#                 f"Player Number Custom Favorites:{st.session_state.player_id_custom_favorites_file}")
#             if st.session_state.player_id_favorites is not None:
#                 st.write(st.session_state.player_id_favorites)
#             if st.session_state.debug_favorites is not None:
#                 st.write(st.session_state.debug_favorites)

#         with help:
#             if st.session_state.help is None:
#                 st.write('Help not available.')
#             else:
#                 st.markdown(st.session_state.help)

#         with release_notes:
#             if st.session_state.release_notes is None:
#                 st.write('No release notes available.')
#             else:
#                 st.markdown(st.session_state.release_notes)

#         with about:
#             content = slash_about()
#             st.write(content)
#             app_info()

#         with debug:
#             st.header('Debug')
#             st.write('Not yet implemented.')

#     print_to_log_info('create_tab_bar time:', time.time()-t)


# def read_favorites():

#     if st.session_state.default_favorites_file.exists():
#         with open(st.session_state.default_favorites_file, 'r') as f:
#             favorites = json.load(f)
#             st.session_state.favorites = favorites

#     if st.session_state.player_id_custom_favorites_file.exists():
#         with open(st.session_state.player_id_custom_favorites_file, 'r') as f:
#             player_id_favorites = json.load(f)
#             st.session_state.player_id_favorites = player_id_favorites

#     if st.session_state.debug_favorites_file.exists():
#         with open(st.session_state.debug_favorites_file, 'r') as f:
#             debug_favorites = json.load(f)
#             st.session_state.debug_favorites = debug_favorites


# def get_vetted_prompts_from_favorites(favorites, category='Summarize'):
#     # Navigate the JSON path to get the appropriate list of prompts
#     vetted_prompts = [favorites['SelectBoxes']['Vetted_Prompts'][p[1:]] for p in favorites["Buttons"][category]['prompts']]
    
#     return vetted_prompts


def read_configs() -> None:

    st.session_state.default_favorites_file = pathlib.Path(
        'default.favorites.json')
    st.session_state.player_id_custom_favorites_file = pathlib.Path(
        f'favorites/{st.session_state.player_id}.favorites.json')
    st.session_state.debug_favorites_file = pathlib.Path(
        'favorites/debug.favorites.json')

    if st.session_state.default_favorites_file.exists():
        with open(st.session_state.default_favorites_file, 'r') as f:
            favorites = json.load(f)
        st.session_state.favorites = favorites
        #st.session_state.vetted_prompts = get_vetted_prompts_from_favorites(favorites)
    else:
        raise FileNotFoundError(f"Required configuration file not found: {st.session_state.default_favorites_file}.")

    if st.session_state.player_id_custom_favorites_file.exists():
        with open(st.session_state.player_id_custom_favorites_file, 'r') as f:
            player_id_favorites = json.load(f)
        st.session_state.player_id_favorites = player_id_favorites
    else:
        st.session_state.player_id_favorites = None

    if st.session_state.debug_favorites_file.exists():
        with open(st.session_state.debug_favorites_file, 'r') as f:
            debug_favorites = json.load(f)
        st.session_state.debug_favorites = debug_favorites
    else:
        st.session_state.debug_favorites = None

    # display missing prompts in favorites
    if 'missing_in_summarize' not in st.session_state:
        # Get the prompts from both locations
        summarize_prompts = st.session_state.favorites['Buttons']['Summarize']['prompts']
        vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']

        # Process the keys to ignore leading '@'
        st.session_state.summarize_keys = {p.lstrip('@') for p in summarize_prompts}
        st.session_state.vetted_keys = set(vetted_prompts.keys())

        # Find items in summarize_prompts but not in vetted_prompts. There should be none.
        st.session_state.missing_in_vetted = st.session_state.summarize_keys - st.session_state.vetted_keys
        assert len(st.session_state.missing_in_vetted) == 0, f"Oops. {st.session_state.missing_in_vetted} not in {st.session_state.vetted_keys}."

        # Find items in vetted_prompts but not in summarize_prompts. ok if there's some missing.
        st.session_state.missing_in_summarize = st.session_state.vetted_keys - st.session_state.summarize_keys

        print("\nItems in Vetted_Prompts but not in Summarize.prompts:")
        for item in st.session_state.missing_in_summarize:
            print(f"- {item}: {vetted_prompts[item]['title']}")
    return


# def initialize_new_player():

#     content = slash_about()
#     streamlit_chat.message(f"Morty: {content}", key='create_main_section_about_message', logo=st.session_state.assistant_logo)
#     if st.session_state.show_sql_query:
#         streamlit_chat.message(
#             f"Morty: Press the **Summarize** or **AI Predictions** button in the left sidebar or ask me questions using the **prompt box below**. Queries take about 10 seconds to complete.", key='chat_messages_user_no_messages', logo=st.session_state.assistant_logo)
#     else:
#         streamlit_chat.message(
#             f"Morty: Press the **Summarize** or **AI Predictions** button in the left sidebar.", key='chat_messages_user_no_messages', logo=st.session_state.assistant_logo)

#     #assert isinstance(player_id_l, list) and len(
#     #    player_id_l) == 1, player_id_l
#     #player_id = player_id_l[0]
#     if not change_game_state(st.session_state.player_id, None):
#         st.stop()
        
#     create_main_section()


# def create_main_section():

#     # using streamlit's st.chat_input because it stays put at bottom, chat.openai.com style. # was chat_input
#     if st.session_state.show_sql_query:
#         if user_content := st.chat_input("Type your prompt here.", key='user_prompt_input'):
#             ask_a_question_with_context(user_content) # don't think DEFAULT_LARGE_AI_MODEL is needed?
#     # output all messages except the initial system message.
#     # only system message
#     t = time.time()
#     if len(st.session_state.messages) == 1:
#         # todo: put this message into config file.
#         content = slash_about()
#         streamlit_chat.message(f"Morty: {content}", key='create_main_section_about_message', logo=st.session_state.assistant_logo)
#         if st.session_state.show_sql_query:
#             streamlit_chat.message(
#                 f"Morty: Press the **Summarize** or **AI Predictions** button in the left sidebar or ask me questions using the **prompt box below**. Queries take about 10 seconds to complete.", key='chat_messages_user_no_messages', logo=st.session_state.assistant_logo)
#         else:
#             streamlit_chat.message(
#                 f"Morty: Press the **Summarize** or **AI Predictions** button in the left sidebar.", key='chat_messages_user_no_messages', logo=st.session_state.assistant_logo)
#     else:
#         with st.container():

#             pdf_assets = []
#             pdf_assets.append(f"# Bridge Game Postmortem Report Personalized for {st.session_state.player_id}")
#             pdf_assets.append(f"### Created by http://postmortem.chat")
#             pdf_assets.append(f"## Game Date: {st.session_state.game_date} Game ID: {st.session_state.session_id}")
#             print_to_log_info('messages: len:', len(st.session_state.messages))
#             for i, message in enumerate(st.session_state.messages):
#                 if message["role"] == "system":
#                     assert i == 0, "First message should be system message."
#                     continue
#                 if message["role"] == "user":
#                     if st.session_state.show_sql_query:
#                         streamlit_chat.message(
#                             f"You: {message['content']}", is_user=True, key='chat_messages_user_'+str(i))
#                         pdf_assets.append(f"You: {message['content']}")
#                     user_prompt_help = ''
#                     for k, prompt_sqls in st.session_state.vetted_prompts.items():
#                         for prompt_sql in prompt_sqls['prompts']:
#                             if message["content"] == prompt_keyword_replacements(prompt_sql['prompt']) or (prompt_sql['prompt'] == '' and message["content"] == prompt_keyword_replacements(prompt_sql['sql'])):
#                                 user_prompt_help = prompt_sqls['help']
#                                 break
#                     continue
#                 elif message["role"] == "assistant":
#                     # ```sql\nSELECT board_number, contract, COUNT(*) AS frequency\nFROM self\nGROUP BY board_number, contract\nORDER BY board_number, frequency DESC;\n```
#                     if message['content'].startswith('/'):
#                         slash_command = message['content'].split(
#                             ' ', maxsplit=1)
#                         assert len(slash_command), slash_command
#                         if slash_command[0] == '/about':
#                             # ['/about',content]
#                             assert len(slash_command) == 2
#                             streamlit_chat.message(
#                                 f"Morty: {slash_command[1]}", key='main.slash.'+str(i), logo=st.session_state.assistant_logo)
#                             pdf_assets.append(f"🥸 Morty: {slash_command[1]}")
#                         continue
#                     match = re.match(
#                         r'```sql\n(.*)\n```', message['content'])
#                     print_to_log_info('message content:', message['content'], match)
#                     if match is None:
#                         # for unknown reasons, the sql query is returned in 'content'.
#                         # hoping this is a SQL query
#                         sql_query = message['content']
#                         streamlit_chat.message(f"Morty: Oy, invalid SQL query: {sql_query}",
#                                             key='main.invalid.'+str(i), logo=st.session_state.assistant_logo)
#                         pdf_assets.append(f"🥸 Morty: Oy, invalid SQL query: {sql_query}")
#                         continue
#                     else:
#                         # for unknown reasons, the sql query is returned embedded in a markdown code block.
#                         sql_query = match.group(1)
#                 elif message['role'] == 'function':
#                     sql_query = message['content']
#                 else:
#                     assert False, message['role']
#                 if st.session_state.show_sql_query:
#                     streamlit_chat.message(f"Ninja Coder: {sql_query}",
#                                         key='main.embedded_sql.'+str(i), logo=st.session_state.guru_logo)
#                     pdf_assets.append(f"Ninja Coder: {sql_query}")
#                 # use sql query as key. get last dataframe in list.
#                 assert len(
#                     st.session_state.dataframes[sql_query]) > 0, "No dataframes for sql query."
#                 df = st.session_state.dataframes[sql_query][-1]
#                 if df.shape < (1, 1):
#                     assistant_content = f"{user_prompt_help} -- Never happened."
#                     streamlit_chat.message(
#                         f"Morty: {assistant_content}", key='main.empty_dataframe.'+str(i), logo=st.session_state.assistant_logo)
#                     pdf_assets.append(f"🥸 Morty: {assistant_content}")
#                     continue
#                 if df.shape == (1, 1):
#                     assistant_answer = str(df.columns[0]).replace(
#                         'count_star()', 'count').replace('_', ' ')
#                     assistant_scaler = df[0][0]
#                     if assistant_scaler is pd.NA:
#                         assistant_content = f"{assistant_answer} is None."
#                     else:
#                         assistant_content = f"{assistant_answer} is {assistant_scaler}."
#                     streamlit_chat.message(
#                         f"Morty: {user_prompt_help} {assistant_content}", key='main.dataframe_is_scaler.'+str(i), logo=st.session_state.assistant_logo)
#                     pdf_assets.append(f"🥸 Morty: {user_prompt_help} {assistant_content}")
#                     continue
#                 assistant_content = f"{user_prompt_help} Result is a dataframe of {len(df)} rows."
#                 streamlit_chat.message(
#                     f"Morty: {assistant_content}", key='main.dataframe.'+str(i), logo=st.session_state.assistant_logo)
#                 pdf_assets.append(f"🥸 Morty: {assistant_content}")
#                 #df.index.name = 'Row'
#                 st.session_state.df_unique_id += 1 # only needed because message dataframes aren't being released for some unknown reason.
#                 ShowDataFrameTable(
#                     df, key='main_messages_df_'+str(st.session_state.df_unique_id), color_column=None if len(df.columns) <= 1 else df.columns[1], tooltips=st.session_state.dataframe_tooltips) # only colorize if more than one column.
#                 pdf_assets.append(df)
#                 # else:
#                 #    st.dataframe(df.T.style.format(precision=2, thousands=""))

#             if st.session_state.pdf_link.download_button(label="Download Personalized Report",
#                     data=streamlitlib.create_pdf(pdf_assets, title=f"Bridge Game Postmortem Report Personalized for {st.session_state.player_id}"),
#                     file_name = f"{st.session_state.session_id}-{st.session_state.player_id}-morty.pdf",
#                     mime='application/octet-stream'):
#                 st.warning('Personalized report downloaded.')
#             #pdf_base64_encoded = streamlitlib.create_pdf(pdf_assets)
#             #download_pdf_html = f'<a href="data:application/octet-stream;base64,{pdf_base64_encoded.decode()}" download="{st.session_state.session_id}-{st.session_state.player_id}-morty.pdf">Download Personalized Report</a>'
#             #st.session_state.pdf_link.markdown(download_pdf_html, unsafe_allow_html=True) # pdf_link is really a previously created st.sidebar.empty().

#     # wish this would scroll to top of page but doesn't work.
#     # js = '''
#     # <script>
#     #     var body = window.parent.document.querySelector(".main");
#     #     console.log(body);
#     #     body.scrollTop = 0;
#     # </script>
#     # '''
#     # st.components.v1.html(js)

#     streamlitlib.move_focus()
#     print_to_log_info('create_main_section time:', time.time()-t)


# def main():

# #     AvatarStyle = [
# #     "adventurer",
# #     "adventurer-neutral",
# #     "avataaars",
# #     "avataaars-neutral",
# #     "big-ears",
# #     "big-ears-neutral",
# #     "big-smile",
# #     "bottts",
# #     "bottts-neutral",
# #     "croodles",
# #     "croodles-neutral",
# #     "fun-emoji",
# #     "icons",
# #     "identicon",
# #     "initials",
# #     "lorelei",
# #     "lorelei-neutral",
# #     "micah",
# #     "miniavs",
# #     "open-peeps",
# #     "personas",
# #     "pixel-art",
# #     "pixel-art-neutral",
# #     "shapes",
# #     "thumbs",
# # ]
# #     for a in AvatarStyle:
# #         streamlit_chat.message(
# #             f"Hi. I'm Morty. Your friendly postmortem chatbot."+a, key='vacation_message_1'+a, avatar_style=a)

#     # first time through
#     if "player_id" not in st.session_state:

#         # initialize values which will never change
#         st.set_page_config(page_title="Morty", page_icon=":robot_face:", layout="wide")
#         #streamlitlib.widen_scrollbars() # removed because it was causing a problem for Aggrid table height.
#         st.session_state.app_datetime = datetime.fromtimestamp(pathlib.Path(
#             __file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
#         # in case there's no ai_apis.json file
#         import platform
#         if platform.system() == 'Windows': # ugh. this hack is required because torch somehow remembers the platform where the model was created. Must be a bug. Must lie to torch.
#             pathlib.PosixPath = pathlib.WindowsPath
#         else:
#             pathlib.WindowsPath = pathlib.PosixPath
#         st.session_state.main_section_container = st.empty()
#         st.session_state.pdf_assets = []
#         #st.session_state.ai_api = DEFAULT_AI_MODEL # feature removed
#         st.session_state.con = duckdb.connect()
#         st.session_state.con_register_name = 'self'
#         st.session_state.show_sql_query = os.getenv('STREAMLIT_ENV') == 'development'
#         st.session_state.sql_query_mode = False
#         st.session_state.sql_queries = []
#         st.session_state.player_id = None
#         st.session_state.session_id = None
#         st.session_state.game_urls_d = {}
#         st.session_state.tournament_session_urls_d = {}
#         st.session_state.single_dummy_sample_count = 10
#         st.session_state.do_not_cache_df = True

#         #st.session_state.df_unique_id = 0 # only needed because message dataframes aren't being released for some unknown reason.
#         #st.session_state.assistant_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_assistant.gif?raw=true' # 🥸 todo: put into config. must have raw=true for github url.
#         #st.session_state.guru_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_guru.png?raw=true' # 🥷todo: put into config file. must have raw=true for github url.
#         st.session_state.assistant_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_assistant.gif?raw=true' # 🥸 todo: put into config. must have raw=true for github url.
#         st.session_state.guru_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_guru.png?raw=true' # 🥷todo: put into config file. must have raw=true for github url.

#         # causes streamlit connection error
#         # if os.environ.get('STREAMLIT_ENV') is not None and os.environ.get('STREAMLIT_ENV') == 'development':
#         #     if os.environ.get('STREAMLIT_QUERY_STRING') is not None:
#         #         # todo: need to parse STREAMLIT_QUERY_STRING instead of hardcoding.
#         #         if 'player_id' not in st.query_params:
#         #             obsolete? st.experimental_set_query_params(player_id=2663279)
#         # http://localhost:8501/?player_id=2663279

#     if 'player_id' in st.query_params:
#         player_id = st.query_params['player_id']
#         if not isinstance(player_id, str):
#             st.error(f'player_id must be a string {player_id}')
#             st.stop()
#         st.session_state.player_id = player_id

#     if st.session_state.player_id is None:
#         st.sidebar.caption(f"App:{st.session_state.app_datetime}")
#         if False:
#             streamlit_chat.message(
#                 "Hi. I'm Morty. Your friendly postmortem chatbot.", key='vacation_message_1', logo=st.session_state.assistant_logo)
#             streamlit_chat.message(
#                 "I'm on a well deserved vacation while my overlord swaps out my chat API for something more economically sustainable. Should be back in a week or so. Meanwhile, happy prompting.", key='vacation_message_2', logo=st.session_state.assistant_logo)
#             app_info()
#             st.stop()
#         st.sidebar.text_input(
#             "Enter an ACBL player number", on_change=player_id_change, placeholder='2663279', key='player_id_input')
#         st.session_state.main_section_container = st.container()
#         with st.session_state.main_section_container:
#             streamlit_chat.message(
#                 "Hi. I'm Morty. Your friendly postmortem chatbot. I only want to chat about ACBL pair matchpoint games using a Mitchell movement and not shuffled.", key='intro_message_1', logo=st.session_state.assistant_logo)
#             streamlit_chat.message(
#                 "I'm optimized for large screen devices such as a notebook or monitor. Do not use a smartphone.", key='intro_message_2', logo=st.session_state.assistant_logo)
#             streamlit_chat.message(
#                 "To start our postmortem chat, I'll need an ACBL player number. I'll use it to find player's latest ACBL club game. It will be the subject of our chat.", key='intro_message_3', logo=st.session_state.assistant_logo)
#             streamlit_chat.message(
#                 "Enter any ACBL player number in the left sidebar.", key='intro_message_4', logo=st.session_state.assistant_logo)
#             streamlit_chat.message(
#                 "I'm just a Proof of Concept so don't double me.", key='intro_message_5', logo=st.session_state.assistant_logo)
#             app_info()

#     else:
#         create_ui()

def initialize_website_specific() -> None:

    st.session_state.assistant_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_assistant.gif?raw=true' # 🥸 todo: put into config. must have raw=true for github url.
    st.session_state.guru_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_guru.png?raw=true' # 🥷todo: put into config file. must have raw=true for github url.
    st.session_state.game_results_url_default = None
    st.session_state.game_name = 'acbl'
    st.session_state.game_results_url = st.session_state.game_results_url_default
    st.session_state.rootPath = pathlib.Path('e:/bridge/data')
    if st.session_state.rootPath.exists():
        st.session_state.acblPath = st.session_state.rootPath.joinpath('acbl')
        st.session_state.savedModelsPath = st.session_state.acblPath.joinpath('SavedModels')
    else:
        st.session_state.rootPath = pathlib.Path('.')
        if not st.session_state.rootPath.exists():
            st.error(f'rootPath does not exist: {st.session_state.rootPath}')
        st.session_state.savedModelsPath = st.session_state.rootPath.joinpath('SavedModels')
    streamlit_chat.message(
        "Hi. I'm Morty. Your friendly postmortem chatbot. I only want to chat about ACBL pair matchpoint games using a Mitchell movement.", key='intro_message_1', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "I'm optimized for large screen devices such as a notebook or monitor. Do not use a smartphone.", key='intro_message_2', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "To start our postmortem chat, I'll need an ACBL player number. I'll use it to find player's latest ACBL club game. It will be the subject of our chat.", key='intro_message_3', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "Enter any ACBL player number in the left sidebar.", key='intro_message_4', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "I'm just a Proof of Concept so don't double me.", key='intro_message_5', logo=st.session_state.assistant_logo)
    app_info()
    return


# this version of perform_hand_augmentations_locked() uses self for class compatibility, older versions did not.
def perform_hand_augmentations_queue(self: Any, hand_augmentation_work: Any) -> None:
    return streamlitlib.perform_queued_work(self, hand_augmentation_work, "Hand analysis")



def augment_df(df: Any) -> Any:
    with st.spinner('Augmenting data...'):
        augmenter = AllAugmentations(
            df,
            None,
            sd_productions=st.session_state.single_dummy_sample_count,
            progress=st.progress(0),
            lock_func=perform_hand_augmentations_queue,
        )
        df, hrs_cache_df = augmenter.perform_all_augmentations()
        print(df.select(pl.col(pl.Float64)).columns)
        # todo: create these Float64 as 32: ['lifemaster_n', 'lifemaster_s', 'lifemaster_e', 'lifemaster_w', 'Pct_NS', 'Pct_EW', 'Declarer_Rating', 'Declarer_Pct']
        df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32)) # todo: this should be done earlier.
    # previously_missing_columns = {'CT_EW_S_Game', 'HCP_NS_S', 'CT_EW_C_Partial', 'CT_NS_C_GSlam', 'CT_NS_C_SSlam', 'CT_NS_D', 'CT_NS_C', 'QT_NS_S', 'CT_EW_H', 'CT_EW_H_GSlam', 'CT_NS_N_SSlam', 'LoTT_Variance', 'MP_Sum_EW', 'CT_NS_C_Pass', 'MP_Geo_NS', 'CT_EW_N', 'CT_EW_N_Pass', 'DP_EW_S', 'iPlayer_Number_W', 'CT_EW_C', 'MP_Geo_EW', 'QT_EW_D', 'QT_NS_D', 'iPlayer_Number_N', 'CT_EW_C_Pass', 'CT_NS_S_Partial', 'CT_NS_H_Pass', 'CT_EW_H_Partial', 'HCP_EW_C', 'CT_EW_N_Game', 'CT_NS_S_SSlam', 'HCP_NS_H', 'CT_NS_D_SSlam', 'CT_NS_D_Pass', 'CT_NS_C_Partial', 'CT_NS_S_Pass', 'HCP_EW_D', 'CT_EW_N_SSlam', 'CT_EW_C_Game', 'HCP_EW_H', 'CT_EW_S_Partial', 'CT_NS_N', 'QT_EW_S', 'CT_EW_S', 'mp_total_n', 'HCP_EW_S', 'CT_NS_C_Game', 'CT_EW_S_GSlam', 'CT_NS_H_Partial', 'iPlayer_Number_S', 'CT_NS_S_GSlam', 'CT_NS_N_GSlam', 'DP_NS_H', 'CT_NS_H_GSlam', 'CT_EW_S_SSlam', 'iPlayer_Number_E', 'CT_NS_H_SSlam', 'DP_EW_C', 'mp_total_w', 'CT_EW_H_Pass', 'CT_NS_D_Game', 'CT_EW_D_GSlam', 'LoTT_Suit_Length', 'CT_EW_C_SSlam', 'QT_EW_H', 'CT_NS_H_Game', 'CT_NS_H', 'QT_NS_H', 'CT_NS_D_GSlam', 'DP_NS_D', 'MP_Sum_NS', 'CT_EW_H_Game', 'DP_NS_C', 'CT_NS_D_Partial', 'CT_EW_S_Pass', 'CT_NS_N_Game', 'CT_NS_N_Pass', 'CT_EW_D_Pass', 'QT_EW_C', 'ParScore_NS', 'HCP_NS_C', 'CT_EW_H_SSlam', 'CT_EW_D_Game', 'HCP_NS_D', 'mp_total_s', 'CT_EW_N_GSlam', 'CT_NS_S', 'CT_EW_D_SSlam', 'CT_EW_D_Partial', 'CT_EW_D', 'CT_EW_C_GSlam', 'DP_EW_D', 'LoTT_Tricks', 'CT_EW_N_Partial', 'DP_NS_S', 'mp_total_e', 'CT_NS_S_Game', 'CT_NS_N_Partial', 'QT_NS_C', 'DP_EW_H'}
    # now_missing_cols = previously_missing_columns.difference(df.columns)
    # print(now_missing_cols)
    # df = df.with_columns([
    #     pl.lit(None).alias(col) for col in now_missing_cols
    # ])
    #assert not now_missing_cols, now_missing_cols
    # with st.spinner('Creating hand data...'):
    #     augmenter = HandAugmenter(df,{},sd_productions=st.session_state.single_dummy_sample_count,progress=st.progress(0),lock_func=perform_hand_augmentations_queue)
    #     df = augmenter.perform_hand_augmentations()
    # with st.spinner('Augmenting with result data...'):
    #     augmenter = ResultAugmenter(df,{})
    #     df = augmenter.perform_result_augmentations()
    # with st.spinner('Augmenting with contract data...'):
    #     augmenter = ScoreAugmenter(df)
    #     df = augmenter.perform_score_augmentations()
    # with st.spinner('Augmenting with DD and SD data...'):
    #     augmenter = DDSDAugmenter(df)
    #     df = augmenter.perform_dd_sd_augmentations()
    # with st.spinner('Augmenting with matchpoints and percentages data...'):
    #     augmenter = MatchPointAugmenter(df)
    #     df = augmenter.perform_matchpoint_augmentations()
    return df


def process_prompt_macros(sql_query: str) -> str:
    replacements = {
        '{Player_Direction}': st.session_state.player_direction,
        '{Partner_Direction}': st.session_state.partner_direction,
        '{Pair_Direction}': st.session_state.pair_direction,
        '{Opponent_Pair_Direction}': st.session_state.opponent_pair_direction
    }
    for old, new in replacements.items():
        if new is None:
            continue
        sql_query = sql_query.replace(old, new)
    return sql_query


def write_report() -> None:
    # bar_format='{l_bar}{bar}' isn't working in stqdm. no way to suppress r_bar without editing stqdm source code.
    # todo: need to pass the Button title to the stqdm description. this is a hack until implemented.
    st.session_state.main_section_container = st.container(border=True)
    with st.session_state.main_section_container:
        report_title = f"Bridge Game Postmortem Report Personalized for {st.session_state.player_name}" # can't use (st.session_state.player_id) because of href link below.
        report_creator = f"Created by https://{st.session_state.game_name}.postmortem.chat"
        report_event_info = f"{st.session_state.game_description} (event id {st.session_state.session_id})."
        report_game_results_webpage = f"Results Page: {st.session_state.game_results_url}"
        report_your_match_info = f"Your pair was {st.session_state.pair_id}{st.session_state.pair_direction} in section {st.session_state.section_name}. You played {st.session_state.player_direction}. Your partner was {st.session_state.partner_name} ({st.session_state.partner_id}) who played {st.session_state.partner_direction}."
        st.markdown('<div style="height: 50px;"><a id="top-of-report" name="top-of-report"></a></div>', unsafe_allow_html=True)
        # Hidden machine-readable metadata for automation tools (e.g. acbl_postmortem_generator.py).
        # The ISO date lets the daily scheduler decide whether the player actually
        # played on a given date before sending out an email.
        game_date_iso = ''
        try:
            gd = st.session_state.get('game_date')
            if gd is not None:
                game_date_iso = gd.isoformat() if hasattr(gd, 'isoformat') else str(gd)
        except Exception:
            game_date_iso = ''
        st.markdown(
            f'<span id="cli-game-meta" '
            f'data-game-date="{game_date_iso}" '
            f'data-player-id="{st.session_state.get("player_id") or ""}" '
            f'data-session-id="{st.session_state.get("session_id") or ""}" '
            f'style="display:none;"></span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### {report_title}")
        st.markdown(f"##### {report_creator}")
        st.markdown(f"#### {report_event_info}")
        st.markdown(f"##### {report_game_results_webpage}")
        st.markdown(f"#### {report_your_match_info}")
        pdf_assets = st.session_state.pdf_assets
        pdf_assets.clear()
        pdf_assets.append(f"# {report_title}")
        pdf_assets.append(f"#### {report_creator}")
        pdf_assets.append(f"### {report_event_info}")
        pdf_assets.append(f"#### {report_game_results_webpage}")
        pdf_assets.append(f"### {report_your_match_info}")
        vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']
        sql_query_count = 0
        prompts_list = st.session_state.selected_button['prompts']
        total = len(prompts_list)
        progress_bar = st.progress(0.0)
        for idx, stats in enumerate(prompts_list, start=1):
            assert stats[0] == '@', stats
            stat = vetted_prompts[stats[1:]]
            for i, prompt in enumerate(stat['prompts']):
                if 'sql' in prompt and prompt['sql']:
                    if i == 0:
                        streamlit_chat.message(f"Morty: {stat['help']}", key=f'morty_sql_query_{sql_query_count}', logo=st.session_state.assistant_logo)
                        pdf_assets.append(f"### {stat['help']}")
                    prompt_sql = prompt['sql']
                    sql_query = process_prompt_macros(prompt_sql)
                    try:
                        query_df = ShowDataFrameTable(None, query=sql_query, key=f'sql_query_{sql_query_count}')
                        if query_df is not None:
                            pdf_assets.append(query_df)
                    except Exception as e:
                        st.error(f"Query failed: {sql_query[:100]}... Error: {e}")
                    sql_query_count += 1
            # update progress
            try:
                progress_bar.progress(min(idx / max(1, total), 1.0))
            except Exception:
                pass
        # clear progress bar at end
        with contextlib.suppress(Exception):
            progress_bar.empty()

        # As a text link
        #st.markdown('[Back to Top](#your-personalized-report)')

        # As an html button (needs styling added)
        # can't use link_button() restarts page rendering. markdown() will correctly jump to href.
        # st.link_button('Go to top of report',url='#your-personalized-report')\
        report_title_anchor = report_title.replace(' ','-').lower()
        st.markdown('''
            <div style="text-align: center; margin: 20px 0;">
                <a href="#top-of-report" style="text-decoration: none;">
                    <button style="padding: 8px 16px; background-color: #ff4b4b; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">
                        Go to top of report
                    </button>
                </a>
            </div>
        ''', unsafe_allow_html=True)

    if st.session_state.pdf_link.download_button(label="Download Personalized Report",
            data=streamlitlib.create_pdf(st.session_state.pdf_assets, title=f"Bridge Game Postmortem Report Personalized for {st.session_state.player_id}"),
            file_name = f"{st.session_state.session_id}-{st.session_state.player_id}-morty.pdf",
            disabled = len(st.session_state.pdf_assets) == 0,
            mime='application/octet-stream',
            key='personalized_report_download_button'):
        st.warning('Personalized report downloaded.')
    return


def ask_sql_query() -> None:

    if st.session_state.show_sql_query:
        with st.container():
            with bottom():
                st.chat_input('Enter a SQL query e.g. SELECT PBN, Contract, Result, N, S, E, W', key='main_prompt_chat_input', on_submit=chat_input_on_submit)


def create_ui() -> None:
    create_sidebar()
    if not st.session_state.sql_query_mode:
        #create_tab_bar()
        if st.session_state.session_id is not None:
            write_report()
    ask_sql_query()


def initialize_session_state() -> None:
    st.set_page_config(layout="wide")
    # Add this auto-scroll code
    streamlitlib.widen_scrollbars()

    if platform.system() == 'Windows': # ugh. this hack is required because torch somehow remembers the platform where the model was created. Must be a bug. Must lie to torch.
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath
    
    if 'player_id' in st.query_params:
        player_id = st.query_params['player_id']
        if not isinstance(player_id, str):
            st.error(f'player_id must be a string {player_id}')
            st.stop()
        st.session_state.player_id = player_id
    else:
        st.session_state.player_id = None

    first_time_defaults = {
        'first_time': True,
        'single_dummy_sample_count': 10,
        'show_sql_query': True, # os.getenv('STREAMLIT_ENV') == 'development',
        'use_historical_data': False,
        'con_register_name': 'self',
        'main_section_container': st.empty(),
        'app_datetime': datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
        'current_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'player_id_default': '2663279',
    }
    for key, value in first_time_defaults.items():
        st.session_state[key] = value

    reset_game_data()
    initialize_website_specific()
    return


def reset_game_data() -> None:

    # Default values for session state variables
    reset_defaults = {
        'game_description_default': None,
        'group_id_default': None,
        'session_id_default': None,
        'section_name_default': None,
        'player_id_default': None,
        'partner_id_default': None,
        'player_name_default': None,
        'partner_name_default': None,
        'player_direction_default': None,
        'partner_direction_default': None,
        'pair_id_default': None,
        'pair_direction_default': None,
        'opponent_pair_direction_default': None,
        'button_title_default': 'Summarize',
    }
    
    # Initialize default values if not already set
    for key, value in reset_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize additional session state variables that depend on defaults.
    reset_session_vars = {
        'df': None,
        'game_description': st.session_state.game_description_default,
        'group_id': st.session_state.group_id_default,
        'session_id': st.session_state.session_id_default,
        'section_name': st.session_state.section_name_default,
        'player_id': st.session_state.player_id_default,
        'partner_id': st.session_state.partner_id_default,
        'player_name': st.session_state.player_name_default,
        'partner_name': st.session_state.partner_name_default,
        'player_direction': st.session_state.player_direction_default,
        'partner_direction': st.session_state.partner_direction_default,
        'pair_id': st.session_state.pair_id_default,
        'pair_direction': st.session_state.pair_direction_default,
        'opponent_pair_direction': st.session_state.opponent_pair_direction_default,
        'button_title': st.session_state.button_title_default,
        #'sidebar_loaded': False,
        'analysis_started': False,   # new flag for analysis sidebar rewrite
        'vetted_prompts': [],
        'pdf_assets': [],
        'sql_query_mode': False,
        'sql_queries': [],
        'game_urls_d': {},
        'tournament_session_urls_d': {},
        'ai_predictions_requested': False,
    }
    
    for key, value in reset_session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value

    return


def app_info() -> None:
    """Display app information"""
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita")
    st.caption(f"App:{st.session_state.app_datetime} Streamlit:{st.__version__} Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}")
    st.caption(f"Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pd.__version__} polars:{pl.__version__} endplay:{endplay.__version__}")
    return


class BridgeGamePostmortemChatbot(PostmortemBase):
    """Bridge Game Postmortem Chatbot Streamlit application."""
    
    def __init__(self):
        super().__init__()
        # App-specific initialization
    
    def initialize_session_state(self):
        """Initialize app-specific session state."""
        # First initialize common session state
        self.initialize_common_session_state()

        # Default player_id placeholder; URL value (if any) is applied below.
        if 'player_id' not in st.session_state:
            st.session_state.player_id = None

        # Chatbot-specific session state (debug_mode now handled by base class)
        # No additional chatbot-specific variables needed at this time

        self.reset_game_data()
        self.initialize_website_specific()

        # Apply URL query parameters AFTER reset_game_data (which wipes
        # player_id/session_id) so they take effect on a fresh load.
        apply_url_params_to_state()

        # If the URL supplied a player_id, eagerly load that player's games so
        # the sidebar and report render without requiring manual input.
        url_player_id = st.session_state.get('player_id')
        loaded_from_url = False
        if url_player_id:
            url_session_id = st.session_state.get('session_id')
            if change_game_state(url_player_id, url_session_id):
                loaded_from_url = True
            else:
                # Invalid player_id in URL - clear it so the sidebar doesn't crash
                st.session_state.player_id = None
                st.session_state.session_id = None

        # Reflect whatever state we ended up in back to the URL.
        sync_url_params_from_state()

        # On the very first run the framework only renders the sidebar (see
        # PostmortemBase.main); when we loaded a session from the URL we want
        # the report to render too, so force one rerun.
        if loaded_from_url:
            st.rerun()
        
    def reset_game_data(self):
        """Reset game data."""
        # First reset common game data
        self.reset_common_game_data()
        # Chatbot-specific reset handled by global function for now
        
    def initialize_website_specific(self):
        """Initialize app-specific components."""
        # Call global function for chatbot-specific initialization
        initialize_website_specific()
    
    def create_sidebar(self):
        """Create app-specific sidebar."""
        # Call global function for chatbot-specific sidebar
        create_sidebar()
    
    def write_report(self):
        """Generate postmortem report using button selection."""
        # Use the global write_report function that respects selected_button
        write_report()

    def main(self):
        """Main entry: same as base, plus Memory footer on the main page."""
        super().main()
        # Memory footer on the main page (same pattern as Elo_Ratings / ffbridge apps).
        st.caption(streamlitlib.get_memory_caption_line(st))


def main() -> None:
    if 'app' not in st.session_state:
        st.session_state.app = BridgeGamePostmortemChatbot()
    st.session_state.app.main()
    return


if __name__ == "__main__":
    main()

