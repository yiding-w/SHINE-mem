from __future__ import annotations

import unittest
from types import SimpleNamespace

from MemoryTest.training.reconstruction_records import build_prefix_cumulative_record
from MemoryTest.training.recurrent_data import TrainingTurn


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


class FixedRng:
    def __init__(self, random_value: float, uniform_value: float = 0.0):
        self.random_value = random_value
        self.uniform_value = uniform_value

    def random(self):
        return self.random_value

    def uniform(self, lower, upper):
        self.asserted_bounds = (lower, upper)
        return self.uniform_value


def make_args(empty_probability: float = 0.25, max_ratio: float = 0.05):
    return SimpleNamespace(
        reconstruction_scope="prefix_cumulative",
        prefix_cumulative_empty_probability=empty_probability,
        prefix_cumulative_max_prefix_ratio=max_ratio,
    )


class PrefixCumulativeRecordsTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.turns = [
            TrainingTurn(turn_id="s1", text="abcdefghijklmnopqrst", qa=()),
            TrainingTurn(turn_id="s2", text="SECOND SESSION", qa=()),
        ]

    def test_nonempty_prefix_is_removed_from_target_and_prediction_is_rejoined(self):
        rng = FixedRng(random_value=0.9, uniform_value=0.05)
        record = build_prefix_cumulative_record(
            self.tokenizer,
            self.turns,
            make_args(),
            rng,
        )

        self.assertEqual(record["session_prefix"], "a")
        self.assertEqual(record["answer"], "bcdefghijklmnopqrst\nSECOND SESSION")
        self.assertEqual(record["reference"], "abcdefghijklmnopqrst\nSECOND SESSION")
        self.assertEqual(record["prediction_prefix"], "a")
        self.assertNotIn("a" * 2, record["answer"][:2])
        self.assertIn("Do not repeat the supplied prefix", record["prompt"])
        self.assertEqual(rng.asserted_bounds, (0.0, 0.05))

    def test_empty_prefix_requires_full_history_reconstruction(self):
        record = build_prefix_cumulative_record(
            self.tokenizer,
            self.turns,
            make_args(),
            FixedRng(random_value=0.1),
        )

        self.assertEqual(record["session_prefix"], "")
        self.assertEqual(record["answer"], record["reference"])
        self.assertEqual(record["source_turn"], 1)
        self.assertEqual(record["source_turn_id"], "s1")


if __name__ == "__main__":
    unittest.main()
