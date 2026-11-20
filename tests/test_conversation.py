import numpy as np

from core.config import NUM_GOALS, NUM_MODULATORS
from core.state import Action, MotivationalState
from llm import conversation
from llm.conversation import MetaMoChatAssistant


class FlakyChat:
    """Chat stub that fails once with a retryable error and records successful turns."""

    def __init__(self):
        self.history = ["previous turn"]
        self.failures_left = 1

    def send_message(self, message):
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("503 UNAVAILABLE")
        self.history.append(message)
        return type("Response", (), {"text": "ok"})()


def test_retryable_error_preserves_chat_history(monkeypatch):
    monkeypatch.setattr(conversation.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        conversation,
        "_create_client_and_chat",
        lambda: (_ for _ in ()).throw(AssertionError("chat must not be recreated")),
    )
    chat = FlakyChat()
    assistant = MetaMoChatAssistant()
    assistant.client, assistant.chat = object(), chat
    action = Action(id="safe_answer", goal_correlations=np.zeros(NUM_GOALS), risk_estimate=0.0)
    state = MotivationalState(G=np.full(NUM_GOALS, 0.5), M=np.full(NUM_MODULATORS, 0.5))

    assert assistant.generate_final_response("hello", action, state) == "ok"
    assert assistant.chat is chat
    assert chat.history[0] == "previous turn"
    assert len(chat.history) == 2
