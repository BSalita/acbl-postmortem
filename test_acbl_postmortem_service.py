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
    def setUp(self) -> None:
        service._SESSION_LISTS.clear()

    def test_latest_club_session_is_ordered_by_game_date_not_session_id(self) -> None:
        games = _club_games(("99620", "2020-02-16"), ("993420", "2024-04-13"), ("985305", "2024-03-28"))
        with mock.patch.object(service.club_api, "player_club_games", return_value=games), \
                mock.patch.object(service.club_api, "player_tournament_sessions", return_value={}):
            sessions = service.list_available_postmortems("2663279")
        self.assertEqual([row["session_id"] for row in sessions], ["993420", "985305", "99620"])


class SessionListCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        service._SESSION_LISTS.clear()

    def test_second_lookup_within_ttl_reuses_the_list(self) -> None:
        games = _club_games(("993420", "2024-04-13"))
        with mock.patch.object(service.club_api, "player_club_games", return_value=games) as club,                 mock.patch.object(service.club_api, "player_tournament_sessions", return_value={}):
            first = service.list_available_postmortems("2663279")
            first[0]["session_id"] = "mutated"
            second = service.list_available_postmortems("2663279")
        self.assertEqual(club.call_count, 1)
        self.assertEqual(second[0]["session_id"], "993420")

    def test_expired_entry_is_fetched_again(self) -> None:
        games = _club_games(("993420", "2024-04-13"))
        with mock.patch.object(service.club_api, "player_club_games", return_value=games) as club,                 mock.patch.object(service.club_api, "player_tournament_sessions", return_value={}):
            service.list_available_postmortems("2663279")
            stamp, rows = service._SESSION_LISTS["2663279"]
            service._SESSION_LISTS["2663279"] = (stamp - service.SESSION_LIST_TTL_S - 1, rows)
            service.list_available_postmortems("2663279")
        self.assertEqual(club.call_count, 2)

    def test_list_built_after_an_api_error_is_not_cached(self) -> None:
        games = _club_games(("993420", "2024-04-13"))
        with mock.patch.object(service.club_api, "player_club_games", return_value=games),                 mock.patch.object(
                    service.club_api,
                    "player_tournament_sessions",
                    side_effect=service.club_api.ClubApiClientError("down"),
                ):
            service.list_available_postmortems("2663279")
        self.assertNotIn("2663279", service._SESSION_LISTS)


class SqlBatchTests(unittest.TestCase):
    def test_batch_keeps_order_and_isolates_a_failing_statement(self) -> None:
        frame = pl.DataFrame({"Board": [1, 2, 3]})
        results = service.run_sql_batch(
            frame,
            ["SELECT COUNT(*) AS n FROM self", "SELECT Missing_Column FROM self", "SELECT MAX(Board) AS m"],
            {},
        )
        self.assertEqual(results[0]["rows"], [{"n": 3}])
        self.assertIn("Missing_Column", results[1]["error"])
        self.assertEqual(results[2]["rows"], [{"m": 3}])
        self.assertTrue(all("elapsed_s" in row for row in results))

    def test_endpoint_loads_once_with_predictions_when_any_statement_needs_them(self) -> None:
        frame = pl.DataFrame({"Board": [1], "Pct_NS_Pred": [0.5]})
        with mock.patch.object(service, "load_postmortem", return_value=(frame, {})) as load:
            response = TestClient(api.app).post(
                "/acbl-postmortem/sql-batch",
                json={
                    "player_id": "2663279",
                    "session_id": "993420",
                    "statements": ["SELECT Board FROM self", "SELECT AVG(Pct_NS_Pred) AS p FROM self"],
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(load.call_count, 1)
        self.assertTrue(load.call_args.kwargs["predictions"])
        self.assertEqual(len(response.json()["results"]), 2)

    def test_empty_batch_is_rejected(self) -> None:
        response = TestClient(api.app).post(
            "/acbl-postmortem/sql-batch",
            json={"player_id": "2663279", "statements": []},
        )
        self.assertEqual(response.status_code, 422)


class EloDirTests(unittest.TestCase):
    def test_elo_dir_comes_from_env(self) -> None:
        import acbl_session_predictions as predictions

        with mock.patch.dict("os.environ", {"ACBL_ELO_DIR": "/data"}):
            self.assertEqual(str(predictions.elo_dir()).replace("\\", "/"), "/data")

    def test_elo_dir_unset_fails_fast(self) -> None:
        import acbl_session_predictions as predictions

        with mock.patch.dict("os.environ", {"ACBL_ELO_DIR": ""}):
            with self.assertRaises(RuntimeError):
                predictions.elo_dir()


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


class RawResultColumnTests(unittest.TestCase):
    def test_result_is_integer_in_sql_when_parquet_has_raw_strings(self) -> None:
        frame = pl.DataFrame(
            {
                "Result": pl.Series([0, 1, -1], dtype=pl.Int8),
                "result": ["=", "+1", "-1"],
                "Tricks": pl.Series([9, 10, 8], dtype=pl.Int16),
                "tricks_taken": ["9", "10", "8"],
            }
        )
        merged = service.club_api._merge_raw_result_columns(frame)
        self.assertEqual(merged.columns, ["Result", "Tricks"])
        result = service.run_sql(merged, "SELECT Result FROM self", {})
        self.assertEqual([row["Result"] for row in result["rows"]], [0, 1, -1])


if __name__ == "__main__":
    unittest.main()
