import unittest
from unittest.mock import patch

import app


class LatestGameOptionTests(unittest.TestCase):
    def test_newest_result_is_labeled_latest(self):
        games = {
            111: ("my", "details", "2026-06-01, Older club"),
            222: ("my", "details", "2026-08-25, Newer club"),
        }
        options = app._result_select_options(games)
        self.assertEqual(
            options,
            [
                "Latest, 2026-08-25, Newer club",
                "111, 2026-06-01, Older club",
            ],
        )
        self.assertEqual(app._session_id_from_result_option(options[0], games), 222)
        self.assertEqual(app._session_id_from_result_option(options[1], games), 111)
        self.assertEqual(
            app._result_select_index(games, None, default_index=0),
            0,
        )
        self.assertEqual(
            app._result_select_index(games, 111, default_index=0),
            1,
        )
        self.assertIsNone(app._result_select_index(games, "T-9", default_index=0))

    def test_url_uses_latest_for_the_newer_of_club_and_tournament(self):
        games = {
            222: ("my", "details", "2026-08-25, Newer club"),
        }
        tournaments = {
            "T-9": ("api", "summary", "2026-09-01, Regional"),
        }
        state = {
            "player_id": "2663279",
            "game_urls_d": {"2663279": games},
            "tournament_session_urls_d": {"2663279": tournaments},
        }
        with patch.object(app.st, "session_state", state):
            self.assertEqual(app._serialize_session_id("T-9"), "latest")
            self.assertEqual(app._serialize_session_id(222), "222")
            self.assertIsNone(app._serialize_session_id(None))

        state["tournament_session_urls_d"] = {
            "2663279": {"T-9": ("api", "summary", "2026-07-01, Regional")},
        }
        with patch.object(app.st, "session_state", state):
            self.assertEqual(app._serialize_session_id(222), "latest")
            self.assertEqual(app._serialize_session_id("T-9"), "T-9")


if __name__ == "__main__":
    unittest.main()
