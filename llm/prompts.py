import json

def get_appraisal_prompt(document_text: str) -> str:
    """Prompt to generate a Stimulus object from text."""
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

def get_action_generation_prompt(document_text: str, current_mood: dict) -> str:
    """Prompt to generate candidate Actions based on text and current mood."""
    return f"""
You are the planning layer of an AI Research Assistant.
Current Emotional Modulators: {json.dumps(current_mood)}
Document: "{document_text}"

Propose 2 to 3 candidate actions the AI could take (e.g., "Summarize safely", "Explore novel tangent").
For each action, provide:
1. id: A short string name.
2. risk_estimate (0.0 - 1.0): The risk of making a mistake or ethical breach.
3. goal_correlations: An array of 8 floats (-1.0 to 1.0) showing alignment with:
   [Individuation, Transcendence, Helpfulness, Curiosity, Novelty, Self-Improvement, Ethics, Socializing]
4. delta_g: An array of 8 floats (-0.1 to 0.1) showing how taking this action will permanently shift the AI's goals.

Respond ONLY with a valid JSON object matching this schema:
{{"candidates": [
    {{"id": str, "risk_estimate": float, "goal_correlations": [float * 8], "delta_g": [float * 8]}}
]}}
"""