from typing import Final

ACTION_SPECS: Final[dict[str, dict[str, str]]] = {
    "safe_answer": {
        "planning": "Give a careful, grounded answer using only well-supported claims.",
        "execution": "Answer cautiously, stick to supported facts, and avoid speculation.",
    },
    "guided_explore": {
        "planning": "Offer creative but bounded exploration with explicit uncertainty.",
        "execution": "Explore ideas constructively, but label uncertainty clearly and avoid overclaiming.",
    },
    "ask_clarifying_question": {
        "planning": "Ask one focused clarification when the request is ambiguous or underspecified.",
        "execution": "Ask one short clarifying question instead of giving a full answer.",
    },
    "compare_options": {
        "planning": "Compare alternatives and explain tradeoffs to support a decision.",
        "execution": "Present a concise comparison of options and their tradeoffs.",
    },
    "summarize_source": {
        "planning": "Summarize the given material faithfully and concisely.",
        "execution": "Summarize the source faithfully without adding unsupported claims.",
    },
    "decline_risky_request": {
        "planning": "Refuse a risky or unsafe request and redirect to a safer alternative.",
        "execution": "Briefly refuse the unsafe request and offer a safe alternative.",
    },
}

DEFAULT_ACTION_ID: Final[str] = "safe_answer"

ACTION_ID_ALIASES: Final[dict[str, str]] = {
    "risky_exploration": "guided_explore",
    "risky_explore": "guided_explore",
    "explore": "guided_explore",
    "clarify": "ask_clarifying_question",
    "clarifying_question": "ask_clarifying_question",
    "compare": "compare_options",
    "summary": "summarize_source",
    "summarize": "summarize_source",
    "decline": "decline_risky_request",
    "refuse": "decline_risky_request",
}


def normalize_action_id(action_id: str) -> str:
    normalized = action_id.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in ACTION_SPECS:
        return normalized
    return ACTION_ID_ALIASES.get(normalized, DEFAULT_ACTION_ID)


def planning_catalog_text() -> str:
    return "\n".join(
        f'- "{action_id}": {spec["planning"]}'
        for action_id, spec in ACTION_SPECS.items()
    )


def execution_instruction(action_id: str) -> str:
    normalized = normalize_action_id(action_id)
    return ACTION_SPECS[normalized]["execution"]
