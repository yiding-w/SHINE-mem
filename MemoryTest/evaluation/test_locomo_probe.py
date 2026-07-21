from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from MemoryTest.evaluation.locomo_probe import (
    build_session_texts,
    distance_bucket,
    evidence_distance,
    load_locomo_sample,
    score_prediction,
    select_probe_questions,
    select_probe_session_window,
)


class LoCoMoProbeTest(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "sample_id": "conv-test",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"speaker": "A", "text": "I moved to Paris."}],
                "session_1_date_time": "1 January 2024",
                "session_2": [{"speaker": "B", "text": "How is Paris?"}],
                "session_2_date_time": "2 January 2024",
            },
            "qa": [
                {
                    "question": "Where did A move?",
                    "answer": "Paris",
                    "category": 4,
                    "evidence": ["D1:1"],
                },
                {
                    "question": "When did A move?",
                    "answer": "1 January 2024",
                    "category": 2,
                    "evidence": ["D1:1"],
                },
            ],
        }

    def test_load_render_select_and_distance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "locomo.json"
            path.write_text(json.dumps([self.sample]), encoding="utf-8")
            loaded = load_locomo_sample(path, 0)
        sessions = build_session_texts(loaded)
        self.assertEqual([item["session_number"] for item in sessions], [1, 2])
        self.assertIn("I moved to Paris", sessions[0]["text"])
        questions = select_probe_questions(loaded, [2, 4], 1, seed=42)
        self.assertEqual(len(questions), 2)
        self.assertEqual(evidence_distance(questions[0], 2), 1)
        self.assertEqual(distance_bucket(1), "near_0_2")
        window = select_probe_session_window(loaded, [2, 4], max_sessions=1)
        window_questions = select_probe_questions(
            loaded,
            [2, 4],
            questions_per_category=0,
            seed=42,
            allowed_session_numbers=set(window),
        )
        self.assertEqual(window, [1])
        self.assertEqual(len(window_questions), 2)

    def test_short_answer_f1(self):
        question = self.sample["qa"][0]
        self.assertEqual(score_prediction(question, "Paris"), 1.0)
        self.assertEqual(score_prediction(question, "London"), 0.0)


if __name__ == "__main__":
    unittest.main()
