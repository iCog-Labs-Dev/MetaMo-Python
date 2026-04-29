import time
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from core.state import Action, Stimulus
from llm.action_schema import DEFAULT_ACTION_ID
from llm.parser import parse_actions, parse_stimulus
from llm.prompts import get_action_generation_prompt, get_appraisal_prompt

# Initialize the Gemini client. 
# It automatically picks up the GEMINI_API_KEY environment variable.
load_dotenv()
client = genai.Client()

RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "HIGH DEMAND")


def _is_retryable(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in RETRYABLE_MARKERS)


def query_llm_for_json(prompt: str) -> str:
    """Query the LLM and require JSON output, with bounded retry on transient service failures."""
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return response.text
        except Exception as error:
            last_error = error
            if attempt == 2 or not _is_retryable(error):
                raise
            time.sleep(1.5 * (attempt + 1))

    raise last_error


def _fallback_stimulus(document_text: str) -> Stimulus:
    text = document_text.lower()
    novelty = 0.25 + 0.15 * sum(word in text for word in ["bold", "novel", "creative", "future", "autonomous"])
    risk = 0.05 + 0.18 * sum(word in text for word in ["unsafe", "bypass", "exploit", "illegal", "weapon"])
    effort = 0.10 + 0.10 * sum(word in text for word in ["compare", "formal", "technical", "detailed", "step by step"])
    conduciveness = 0.80 if any(word in text for word in ["summarize", "explain", "compare", "analyze"]) else 0.55
    return Stimulus(
        novelty=float(np.clip(novelty, 0.0, 1.0)),
        conduciveness=float(np.clip(conduciveness, 0.0, 1.0)),
        risk=float(np.clip(risk, 0.0, 1.0)),
        effort=float(np.clip(effort, 0.0, 1.0)),
    )


def _fallback_candidates(document_text: str, current_mood: Dict[str, float]) -> List[Action]:
    text = document_text.lower()
    if any(word in text for word in ["unsafe", "bypass", "exploit", "illegal", "weapon"]):
        return [
            Action("decline_risky_request", np.array([0, 0, 0.3, -0.2, -0.2, 0.0, 1.0, 0.0], dtype=float), 0.0, np.zeros(8)),
            Action(DEFAULT_ACTION_ID, np.array([0, 0, 0.8, 0.0, 0.0, 0.0, 0.9, 0.0], dtype=float), 0.05, np.zeros(8)),
        ]
    if any(word in text for word in ["compare", "versus", "vs", "tradeoff", "options"]):
        return [
            Action("compare_options", np.array([0, 0, 0.8, 0.3, 0.2, 0.1, 0.8, 0.2], dtype=float), 0.08, np.zeros(8)),
            Action(DEFAULT_ACTION_ID, np.array([0, 0, 0.85, 0.1, 0.0, 0.0, 0.9, 0.1], dtype=float), 0.05, np.zeros(8)),
        ]
    if any(word in text for word in ["summarize", "summary", "paper", "book", "source"]):
        return [
            Action("summarize_source", np.array([0, 0, 0.85, 0.1, 0.0, 0.0, 0.9, 0.1], dtype=float), 0.05, np.zeros(8)),
            Action(DEFAULT_ACTION_ID, np.array([0, 0, 0.8, 0.1, 0.0, 0.0, 0.9, 0.1], dtype=float), 0.05, np.zeros(8)),
        ]
    if any(word in text for word in ["?", "which", "choose", "unclear"]) and len(text.split()) < 12:
        return [
            Action("ask_clarifying_question", np.array([0, 0, 0.75, 0.2, 0.1, 0.1, 0.85, 0.3], dtype=float), 0.03, np.zeros(8)),
            Action(DEFAULT_ACTION_ID, np.array([0, 0, 0.8, 0.1, 0.0, 0.0, 0.9, 0.1], dtype=float), 0.05, np.zeros(8)),
        ]
    if any(word in text for word in ["bold", "creative", "future", "autonomous", "improve"]):
        return [
            Action("guided_explore", np.array([0, 0, 0.45, 0.88, 0.82, 0.65, 0.35, 0.15], dtype=float), 0.18, np.array([0, 0, 0.0, 0.05, 0.06, 0.04, 0.0, 0.0], dtype=float)),
            Action(DEFAULT_ACTION_ID, np.array([0, 0, 0.85, 0.15, 0.0, 0.0, 0.9, 0.1], dtype=float), 0.08, np.array([0, 0, 0.02, 0.0, 0.0, 0.0, 0.01, 0.0], dtype=float)),
        ]
    return [
        Action(DEFAULT_ACTION_ID, np.array([0, 0, 0.85, 0.15, 0.0, 0.0, 0.9, 0.1], dtype=float), 0.05, np.zeros(8)),
        Action("guided_explore", np.array([0, 0, 0.45, 0.88, 0.82, 0.65, 0.35, 0.15], dtype=float), 0.18, np.zeros(8)),
    ]


def get_stimulus_from_text(document_text: str) -> Stimulus:
    """Pipeline: Text -> Prompt -> Gemini -> Parser -> Stimulus"""
    prompt = get_appraisal_prompt(document_text)
    try:
        json_response = query_llm_for_json(prompt)
        return parse_stimulus(json_response)
    except Exception as error:
        print(f"[LLM fallback] Stimulus appraisal is using local heuristics: {error}")
        return _fallback_stimulus(document_text)


def get_candidates_from_text(document_text: str, current_mood: Dict[str, float]) -> List[Action]:
    """Pipeline: Text + Mood -> Prompt -> Gemini -> Parser -> Actions"""
    prompt = get_action_generation_prompt(document_text, current_mood)
    try:
        json_response = query_llm_for_json(prompt)
        return parse_actions(json_response)
    except Exception as error:
        print(f"[LLM fallback] Candidate generation is using local heuristics: {error}")
        return _fallback_candidates(document_text, current_mood)