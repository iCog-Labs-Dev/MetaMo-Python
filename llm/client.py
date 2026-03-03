import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
# Assuming these are available in your python path
from llm.prompts import get_appraisal_prompt, get_action_generation_prompt
from llm.parser import parse_stimulus, parse_actions
from core.state import Stimulus, Action
from typing import List, Dict

# Initialize the Gemini client. 
# It automatically picks up the GEMINI_API_KEY environment variable.
load_dotenv()
client = genai.Client()

def query_llm_for_json(prompt: str) -> str:
    """Helper to query the Gemini LLM and enforce JSON output."""
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt,
        config=types.GenerateContentConfig(
            # This explicitly forces the model to return valid JSON
            response_mime_type="application/json",
            # Keep temperature low for consistent numerical estimates
            temperature=0.2 
        )
    )
    return response.text

def get_stimulus_from_text(document_text: str) -> Stimulus:
    """Pipeline: Text -> Prompt -> Gemini -> Parser -> Stimulus"""
    prompt = get_appraisal_prompt(document_text)
    json_response = query_llm_for_json(prompt)
    return parse_stimulus(json_response)

def get_candidates_from_text(document_text: str, current_mood: Dict[str, float]) -> List[Action]:
    """Pipeline: Text + Mood -> Prompt -> Gemini -> Parser -> Actions"""
    prompt = get_action_generation_prompt(document_text, current_mood)
    json_response = query_llm_for_json(prompt)
    return parse_actions(json_response)