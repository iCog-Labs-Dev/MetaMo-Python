from applications.research_assistant.llm.parser import parse_perception, parse_stimulus

__all__ = [
    "MetaMoChatAssistant",
    "get_candidates_from_text",
    "get_calibrated_perception_from_text",
    "get_perception_from_text",
    "get_stimulus_from_text",
    "parse_perception",
    "parse_stimulus",
]


def __getattr__(name):
    if name in {
        "get_candidates_from_text",
        "get_calibrated_perception_from_text",
        "get_perception_from_text",
        "get_stimulus_from_text",
    }:
        from applications.research_assistant.llm import client

        return getattr(client, name)
    if name == "MetaMoChatAssistant":
        from applications.research_assistant.llm.conversation import MetaMoChatAssistant

        return MetaMoChatAssistant
    raise AttributeError(name)
