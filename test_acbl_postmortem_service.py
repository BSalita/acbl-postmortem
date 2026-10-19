from __future__ import annotations

import unittest
from unittest import mock

import polars as pl
from fastapi.testclient import TestClient

import acbl_postmortem_api_server as api
import acbl_postmortem_service as service


def _club_games(*rows: tuple[str, str]) -> dict:
    return {
        sid: ("", f"https://my.acbl.org/club-results/details/{sid}", f"{sid}, {date}, Club, Pairs, 50.00%")
        for sid, date in rows
    }


class LatestSessionTests(unittest.TestCase):
    def test_latest_club_session_is_ordered_by_game_date_not_session_id(self) -> None:
        games = _club_games(("99620", "2020-02-16"), ("993420", "2024-04-13"), ("985305", "2024-03-28"))
        with mock.patch.object(service.club_api, "player_club_games", return_value=games), \
                mock.patch.object(service.club_api, "player_tournament_sessions", return_value={}):
            sessions = service.list_available_postmortems("2663279")
        self.assertEqual([row["session_id"] for row in sessions], ["993420", "985305", "99620"])


class PredictionColumnTests(unittest.TestCase):
    def test_prediction_favorites_need_predictions(self) -> None:
        for sql in (
            "SELECT AVG(Pct_NS_Pred) FROM self",
            "SELECT Pct_NS_Actual FROM self",
            "SELECT AVG(Pct_EW_Pred_Error) FROM self",
            "SELECT ANY_VALUE(Elo_R_NS_EventStart) FROM self",
            "SELECT Declarer_Direction_Pred FROM self",
        ):
            self.assertTrue(service.needs_predictions(sql), sql)

    def test_ordinary_favorites_do_not(self) -> None:
        for sql in ("SELECT Board, Contract, Pct_NS FROM self", "SELECT Predicted FROM self"):
            self.assertFalse(service.needs_predictions(sql), sql)

    def test_session_kind_matches_the_unified_api(self) -> None:
        self.assertEqual(service.session_kind("993420"), "club")
        self.assertEqual(service.session_kind("2402004-0101-1"), "tournament")


class SqlErrorTests(unittest.TestCase):
    def test_unknown_column_is_a_readable_422(self) -> None:
        frame = pl.DataFrame({"Board": [1]})
        with mock.patch.object(service, "load_postmortem", return_value=(frame, {})):
            response = TestClient(api.app).post(
                "/acbl-postmortem/sql",
                json={"player_id": "2663279", "sql": "SELECT Missing_Column FROM self"},
            )
        self.assertEqual(response.status_code, 422)
        self.assertIn("Missing_Column", response.json()["detail"])

    def test_prediction_sql_asks_for_predictions(self) -> None:
        frame = pl.DataFrame({"Pct_NS_Pred": [0.5]})
        with mock.patch.object(service, "load_postmortem", return_value=(frame, {})) as load:
            response = TestClient(api.app).post(
                "/acbl-postmortem/sql",
                json={"player_id": "2663279", "sql": "SELECT AVG(Pct_NS_Pred) AS p FROM self"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(load.call_args.kwargs["predictions"])


if __name__ == "__main__":
    unittest.main()
