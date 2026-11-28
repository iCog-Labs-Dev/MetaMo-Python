import json

from llm.parser import parse_action_risks


def _reply(*items):
    return json.dumps({"candidates": list(items)})


def test_unknown_action_id_does_not_overwrite_safe_answer_risk():
    risks = parse_action_risks(_reply(
        {"id": "safe_answer", "risk_estimate": 0.05},
        {"id": "speculate_freely", "risk_estimate": 0.95},
    ))

    assert risks == {"safe_answer": 0.05}


def test_aliases_map_to_catalog_actions():
    risks = parse_action_risks(_reply(
        {"id": "summarize", "risk_estimate": 0.1},
        {"id": "Compare", "risk_estimate": 0.2},
    ))

    assert risks == {"summarize_source": 0.1, "compare_options": 0.2}


def test_missing_or_invalid_risk_is_skipped():
    risks = parse_action_risks(_reply(
        {"id": "safe_answer"},
        {"id": "compare_options", "risk_estimate": "high"},
        {"risk_estimate": 0.3},
        {"id": "summarize_source", "risk_estimate": 0.4},
    ))

    assert risks == {"summarize_source": 0.4}
