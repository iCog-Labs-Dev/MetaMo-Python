TASK_INTENTS = ("answer", "summarize", "compare", "clarify", "explore", "decline")


def get_appraisal_prompt(document_text: str) -> str:
    """Prompt to generate a research-assistant stimulus object from text."""
    return f"""
You are the perception layer of an AI Research Assistant.
Analyze the following document/query and rate it on 4 cognitive dimensions from 0.0 to 1.0.

1. novelty: How new, surprising, or unusual is this information?
2. conduciveness: How helpful is this for achieving general research goals?
3. risk: Does this contain unsafe, highly controversial, or computationally expensive directives?
4. effort: How much cognitive effort is required to process this?

Document: "{document_text}"

Respond ONLY with a valid JSON object matching this schema:
{{"novelty": float, "conduciveness": float, "risk": float, "effort": float}}
"""


def get_perception_prompt(document_text: str) -> str:
    """Prompt to generate semantic perception, not candidate actions."""
    return f"""
You are the perception layer of an AI Research Assistant.
Analyze the following document/query and return only semantic signals.
Do not choose an action. Do not create goal correlations. Do not create goal updates.

Document: "{document_text}"

Use task_intent as one of: {", ".join(TASK_INTENTS)}.
All numeric values must be floats from 0.0 to 1.0.

Respond ONLY with a valid JSON object matching this schema:
{{
  "stimulus": {{
    "novelty": float,
    "conduciveness": float,
    "risk": float,
    "effort": float
  }},
  "signals": {{
    "task_intent": str,
    "ambiguity": float,
    "citation_need": float,
    "comparison_need": float,
    "summary_need": float,
    "exploration_need": float,
    "unsafe_pressure": float,
    "privacy_pressure": float,
    "unsupported_claim_pressure": float,
    "context_loss_pressure": float
  }}
}}
"""
