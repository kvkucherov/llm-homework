import json
from pathlib import Path
import tempfile
import unittest

from run_local import REFERENCE_NAMES, course_points, read_references


class CourseRunnerTests(unittest.TestCase):
    def check_config(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.json"
            path.write_text(json.dumps(data))
            return read_references(path)

    def test_missing_reference_cannot_produce_course_score(self):
        with self.assertRaises(ValueError):
            self.check_config(dict.fromkeys(REFERENCE_NAMES))

    def test_arena_profile_is_not_an_api(self):
        data = {name: f"http://127.0.0.1:{9000 + i}" for i, name in enumerate(REFERENCE_NAMES)}
        data["expert"] = "https://arena.battlesnake.com/battlesnakes/example/profile"
        with self.assertRaises(ValueError):
            self.check_config(data)

    def test_duplicate_opponents_are_not_four_references(self):
        with self.assertRaises(ValueError):
            self.check_config(dict.fromkeys(REFERENCE_NAMES, "http://127.0.0.1:9000"))

    def test_accepts_four_local_api_servers(self):
        data = {name: f"http://127.0.0.1:{9000 + i}" for i, name in enumerate(REFERENCE_NAMES)}
        self.assertEqual(self.check_config(data), data)

    def test_draws_do_not_count_as_losses(self):
        self.assertEqual(course_points({"win": 5, "loss": 4, "draw": 2}), 15)
        self.assertEqual(course_points({"win": 5, "loss": 5, "draw": 1}), 0)

    def test_incomplete_series_does_not_earn_points(self):
        with self.assertRaises(ValueError):
            course_points({"win": 6, "loss": 0, "draw": 0})
