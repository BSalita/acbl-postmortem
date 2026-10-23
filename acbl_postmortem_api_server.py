"""First-party REST API for ACBL bridge-game postmortems."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import acbl_postmortem_service as service


app = FastAPI(title="ACBL Postmortem API", version="1.0.0")


class SqlRequest(BaseModel):
    player_id: str
    sql: str
    session_id: Optional[str] = None
    limit: int = 500


def _run(callable_, /, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except (FileNotFoundError, KeyError, ValueError, duckdb.Error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    info = _run(service.dataset_info)
    return {"status": "ok", "service": "acbl-postmortem-api", **info}


@app.get("/acbl-postmortem/dataset-info")
def dataset_info() -> dict:
    return _run(service.dataset_info)


@app.get("/acbl-postmortem/sessions")
def sessions(
    player_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    rows = _run(service.list_available_postmortems, player_id)[:limit]
    return {"sessions": rows, "count": len(rows)}


@app.get("/acbl-postmortem/boards")
def boards(
    player_id: str,
    session_id: Optional[str] = Query(None),
    only_my_boards: bool = Query(True),
    columns: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=5000),
) -> dict:
    """Default columns include Par_Contract_NS, Par_Contract_EW, and Is_Sacrifice_Opportunity when the frame has them."""
    frame, meta = _run(service.load_postmortem, player_id, session_id)
    selected = [column.strip() for column in columns.split(",")] if columns else None
    return _run(
        service.board_results,
        frame,
        meta,
        only_my_boards=only_my_boards,
        columns=selected,
        limit=limit,
    )


@app.post("/acbl-postmortem/sql")
def sql(request: SqlRequest) -> dict:
    frame, meta = _run(
        service.load_postmortem,
        request.player_id,
        request.session_id,
        predictions=service.needs_predictions(request.sql),
    )
    result = _run(service.run_sql, frame, request.sql, meta, limit=request.limit)
    result["meta"] = meta
    return result


class SqlBatchRequest(BaseModel):
    player_id: str
    statements: List[str]
    session_id: Optional[str] = None
    limit: int = 500


@app.post("/acbl-postmortem/sql-batch")
def sql_batch(request: SqlBatchRequest) -> dict:
    """Run several statements on one loaded session, in order.

    results[i] answers statements[i]. A failing statement carries "error"
    in its slot and the rest still run. Used by the MCP report jobs.
    """
    if not request.statements:
        raise HTTPException(status_code=422, detail="statements is empty")
    frame, meta = _run(
        service.load_postmortem,
        request.player_id,
        request.session_id,
        predictions=any(service.needs_predictions(sql) for sql in request.statements),
    )
    results = _run(service.run_sql_batch, frame, request.statements, meta, limit=request.limit)
    return {"results": results, "meta": meta}


@app.get("/acbl-postmortem/schema")
def schema(
    player_id: str,
    session_id: Optional[str] = Query(None),
    pattern: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=5000),
) -> dict:
    frame, _meta = _run(service.load_postmortem, player_id, session_id)
    return _run(service.schema_columns, frame, pattern=pattern, limit=limit)


if __name__ == "__main__":
    import uvicorn

    import threading

    import acbl_session_predictions

    # Rating tables take ~5 s to load; warm them while the server starts.
    threading.Thread(
        target=acbl_session_predictions.warm_elo_tables,
        name="warm-elo-tables",
        daemon=True,
    ).start()
    port = int(os.environ.get("ACBL_POSTMORTEM_API_PORT", "8522"))
    print(
        f"[acbl-postmortem-api] start "
        f"{datetime.now(timezone.utc).isoformat()} port={port}",
        flush=True,
    )
    uvicorn.run(app, host="0.0.0.0", port=port)
