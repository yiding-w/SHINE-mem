from ControlSHINE.scripts.convert_counterfact import convert_records


def _record(case_id, subject, true_answer, new_answer, relation_id="P1"):
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "prompt": "{} is located in",
            "relation_id": relation_id,
            "subject": subject,
            "target_true": {"str": true_answer},
            "target_new": {"str": new_answer},
        },
        "paraphrase_prompts": [f"Where is {subject} located?"],
        "neighborhood_prompts": [],
    }


def test_conversion_uses_distinct_same_relation_donor():
    records = [
        _record(1, "Alpha", "France", "Japan"),
        _record(2, "Beta", "Canada", "Brazil"),
    ]
    samples, stats = convert_records(records, seed=1)
    assert stats["written"] == 2
    assert samples[0].context.answer in {"Canada", "Brazil"}
    assert len(
        {
            samples[0].base.answer.casefold(),
            samples[0].memory.answer.casefold(),
            samples[0].context.answer.casefold(),
        }
    ) == 3


def test_conversion_skips_relation_without_third_answer():
    samples, stats = convert_records([_record(1, "Alpha", "France", "Japan")])
    assert samples == []
    assert stats["no_same_relation_third_answer"] == 1


def test_conversion_does_not_mix_relation_ids_with_same_prompt():
    records = [
        _record(1, "Alpha", "France", "Japan", relation_id="P1"),
        _record(2, "Beta", "Canada", "Brazil", relation_id="P2"),
    ]
    samples, stats = convert_records(records)
    assert samples == []
    assert stats["no_same_relation_third_answer"] == 2
