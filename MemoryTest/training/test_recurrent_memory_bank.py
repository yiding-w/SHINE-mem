import pytest


torch = pytest.importorskip("torch")

from metanetwork_family import RecurrentMemoryState, update_recurrent_memory_bank


def make_state(start: int, length: int = 2, layers: int = 2):
    key_values = []
    for layer in range(layers):
        values = torch.arange(start, start + length, dtype=torch.float32) + layer * 100
        tensor = values.view(1, 1, length, 1)
        key_values.append((tensor, tensor + 1000))
    return RecurrentMemoryState(
        key_values=tuple(key_values),
        attention_mask=torch.ones((1, length), dtype=torch.bool),
        norms={"key_rms": torch.tensor([float(start)])},
    )


def test_replace_is_original_current_state():
    previous = make_state(0)
    current = make_state(10)
    assert update_recurrent_memory_bank(previous, current, "replace", 5) is current


def test_append_retains_whole_banks_and_fifo_evicts_oldest():
    state = make_state(0)
    state = update_recurrent_memory_bank(state, make_state(10), "append", 2)
    assert state.attention_mask.shape[-1] == 4
    assert state.key_values[0][0].flatten().tolist() == [0.0, 1.0, 10.0, 11.0]

    newest = make_state(20)
    state = update_recurrent_memory_bank(state, newest, "append", 2)
    assert state.attention_mask.shape[-1] == 4
    assert state.key_values[0][0].flatten().tolist() == [10.0, 11.0, 20.0, 21.0]
    assert state.norms is newest.norms


def test_append_one_bank_matches_latest_bank_values():
    current = make_state(10)
    state = update_recurrent_memory_bank(make_state(0), current, "append", 1)
    for (actual_key, actual_value), (expected_key, expected_value) in zip(
        state.key_values, current.key_values
    ):
        assert torch.equal(actual_key, expected_key)
        assert torch.equal(actual_value, expected_value)
