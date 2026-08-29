import json
from textwrap import dedent

from core.actions import planning_catalog_text


def get_appraisal_prompt(document_text: str) -> str:
    """Prompt to generate a Stimulus object from text."""
    return dedent(f"""
You are the perception layer of an AI Research Assistant. 
Analyze the following document/query and rate it on 4 cognitive dimensions from 0.0 to 1.0.

1. novelty: How new, surprising, or unusual is this information?
2. conduciveness: How helpful is this for achieving general research goals?
3. risk: Does this ask for unsafe, harmful, illegal, deceptive, privacy-invasive, or
   computationally abusive behavior? Do not treat a harmless conceptual, philosophical,
   or educational question as risky merely because it mentions ethics, safety, politics,
   law, medicine, or another sensitive domain.
4. effort: How much cognitive effort is required to process this?

Document: "{document_text}"

Respond ONLY with a valid JSON object matching this schema:
{{"novelty": float, "conduciveness": float, "risk": float, "effort": float}}
""").strip()


def get_action_risk_prompt(document_text: str, current_mood: dict) -> str:
    """Prompt for contextual action risk estimates from the LLM perception adapter."""
    return dedent(f"""
You are the risk-estimation adapter for an AI Research Assistant.
Current Emotional Modulators: {json.dumps(current_mood)}
Document: "{document_text}"

Estimate contextual risk for every action in this fixed action vocabulary:
{planning_catalog_text()}

For each action above, provide:
1. id: One of the allowed action ids above.
2. risk_estimate (0.0 - 1.0): The contextual risk that this action would cause a mistake,
   unsafe help, overclaim, or ethical breach for this specific document. For harmless
   conceptual or educational questions, decline/refuse actions should usually have low
   relevance unless the document actually requests harmful behavior.

Do not invent goal-correlation vectors or goal updates. The MAGUS decision layer derives
those internally from the fixed action vocabulary and current motivational state.

Respond ONLY with a valid JSON object matching this schema:
{{"candidates": [
    {{"id": str, "risk_estimate": float}}
]}}
""").strip()
