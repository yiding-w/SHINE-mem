from __future__ import annotations

import unittest
from types import SimpleNamespace

from MemoryTest.training.reconstruction_records import build_single_session_retention_records
from MemoryTest.training.recurrent_data import TrainingTurn

try:
    import torch

    from MemoryTest.training.shine_train_utils import category_token_losses
except ModuleNotFoundError:  # Local lightweight test environments may not install torch.
    torch = None
    category_token_losses = None


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


class FixedRng:
    def __init__(self, ratio: float):
        self.ratio = ratio

    def uniform(self, lower, upper):
        self.asserted_bounds = (lower, upper)
        return self.ratio


class SingleSessionRetentionTest(unittest.TestCase):
    def test_records_reconstruct_only_the_addressed_session_suffix(self):
        tokenizer = CharacterTokenizer()
        turns = [
            TrainingTurn(turn_id="s1", text="abcdefghij", qa=()),
            TrainingTurn(turn_id="s2", text="ABCDEFGHIJ", qa=()),
        ]
        args = SimpleNamespace(
            completion_prefix_min_ratio=0.2,
            completion_prefix_max_ratio=0.4,
        )
        rng = FixedRng(0.3)

        records = build_single_session_retention_records(tokenizer, turns, args, rng)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["session_prefix"], "abc")
        self.assertEqual(records[0]["answer"], "defghij")
        self.assertEqual(records[0]["reference"], "abcdefghij")
        self.assertEqual(records[1]["session_prefix"], "ABC")
        self.assertEqual(records[1]["answer"], "DEFGHIJ")
        self.assertEqual(records[1]["reference"], "ABCDEFGHIJ")
        self.assertTrue(all(record["loss_reduction"] == "record_mean" for record in records))
        self.assertTrue(all("only the remainder of that same session" in record["prompt"] for record in records))
        self.assertEqual(rng.asserted_bounds, (0.2, 0.4))


class EqualRecordLossTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_record_mean_does_not_give_longer_rows_more_weight(self):
        # Row 0 has one difficult target; row 1 has two easy targets. A token
        # mean weights row 1 twice, while record_mean weights both sessions 1/2.
        logits = torch.zeros((2, 4, 2), dtype=torch.float32)
        logits[0, 0] = torch.tensor([-2.0, 2.0])
        logits[1, 0] = torch.tensor([2.0, -2.0])
        logits[1, 1] = torch.tensor([2.0, -2.0])
        labels = torch.tensor(
            [
                [-100, 0, -100, -100],
                [-100, 0, 0, -100],
            ],
            dtype=torch.long,
        )
        categories = ["reconstruction", "reconstruction"]

        token_mean = category_token_losses(
            logits,
            labels,
            categories,
            ["token_mean", "token_mean"],
        )["reconstruction"]
        record_mean = category_token_losses(
            logits,
            labels,
            categories,
            ["record_mean", "record_mean"],
        )["reconstruction"]

        difficult = torch.nn.functional.cross_entropy(logits[0, 0:1], torch.tensor([0]))
        easy = torch.nn.functional.cross_entropy(logits[1, 0:2], torch.tensor([0, 0]))
        self.assertTrue(torch.allclose(token_mean, (difficult + 2 * easy) / 3))
        self.assertTrue(torch.allclose(record_mean, (difficult + easy) / 2))
        self.assertGreater(record_mean.item(), token_mean.item())


if __name__ == "__main__":
    unittest.main()
