import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False


RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "HIGH DEMAND")
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}

PROVIDER_ADAPTERS = {
    "gemini": "google_genai",
    "google": "google_genai",
    "snet": "openai_compatible",
}

DEFAULT_BASE_URLS = {
    "snet": "https://llm.c.singularitynet.io/v1",
}

PLACEHOLDER_VALUES = {
    "",
    "your_api_key_here",
    "paste_your_api_key_here",
    "your_model_name_here",
    "your_provider_name_here",
}


@dataclass(frozen=True)
class LLMConfig:
    provider_name: str
    adapter_name: str
    model_name: str
    api_key: str
    base_url: str | None
    request_timeout: float


def _load_basic_env(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_environment() -> None:
    load_dotenv()
    _load_basic_env(".env")


_load_environment()


def _clean_env(name: str) -> str:
    return os.getenv(name, "").strip()


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in PLACEHOLDER_VALUES


def _required_env(name: str) -> str:
    value = _clean_env(name)
    if _is_placeholder(value):
        raise RuntimeError(f"Set {name} in .env before calling the LLM provider.")
    return value


def active_provider() -> str:
    """Return the configured provider label from PROVIDER_NAME."""
    return _clean_env("PROVIDER_NAME").lower() or "unconfigured"


def active_model() -> str:
    """Return the configured model label from MODEL_NAME."""
    return _clean_env("MODEL_NAME") or "unconfigured"


def provider_label() -> str:
    """Return a human-readable provider/model label for logs and reports."""
    return f"{active_provider()}:{active_model()}"


def llm_config(provider: str | None = None) -> LLMConfig:
    provider_name = (provider or _required_env("PROVIDER_NAME")).strip().lower()
    adapter_name = PROVIDER_ADAPTERS.get(provider_name)
    if adapter_name is None:
        known = ", ".join(sorted(PROVIDER_ADAPTERS))
        raise ValueError(f"unknown PROVIDER_NAME {provider_name!r}; known providers: {known}")

    model_name = _required_env("MODEL_NAME")
    api_key = _required_env("API_KEY").removeprefix("Bearer ").strip()
    base_url = _clean_env("BASE_URL") or DEFAULT_BASE_URLS.get(provider_name)
    timeout_raw = _clean_env("REQUEST_TIMEOUT") or "90"

    if adapter_name == "openai_compatible" and not base_url:
        raise RuntimeError(
            "OpenAI-compatible provider selected, but BASE_URL is not set. "
            "Set BASE_URL in .env or use a provider with a known default endpoint."
        )

    return LLMConfig(
        provider_name=provider_name,
        adapter_name=adapter_name,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url.rstrip("/") if base_url else None,
        request_timeout=float(timeout_raw),
    )


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError) and error.code in RETRYABLE_HTTP_STATUS:
        return True
    message = str(error).upper()
    return any(marker in message for marker in RETRYABLE_MARKERS)


def _with_retries(call):
    last_error = None
    for attempt in range(3):
        try:
            return call()
        except Exception as error:
            last_error = error
            if attempt == 2 or not _is_retryable(error):
                raise
            time.sleep(1.5 * (attempt + 1))
    raise last_error


def _google_genai_generate(
    config: LLMConfig,
    prompt: str,
    *,
    system_instruction: str | None,
    json_mode: bool,
    temperature: float,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.api_key)
    config_kwargs: dict[str, Any] = {"temperature": temperature}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=config.model_name,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text


def _openai_compatible_generate(
    config: LLMConfig,
    prompt: str,
    *,
    system_instruction: str | None,
    json_mode: bool,
    temperature: float,
) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    if json_mode:
        messages.append(
            {
                "role": "system",
                "content": "Return only valid JSON. Do not use Markdown or explanatory text.",
            }
        )
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": temperature,
    }
    request = urllib.request.Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=config.request_timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content

    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text

    raise RuntimeError("Chat-completions response did not contain text.")


def generate_text(
    prompt: str,
    *,
    system_instruction: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
    provider: str | None = None,
) -> str:
    """Generate text using PROVIDER_NAME, MODEL_NAME, and API_KEY from .env."""
    config = llm_config(provider=provider)

    if config.adapter_name == "google_genai":
        return _with_retries(
            lambda: _google_genai_generate(
                config,
                prompt,
                system_instruction=system_instruction,
                json_mode=json_mode,
                temperature=temperature,
            )
        )
    if config.adapter_name == "openai_compatible":
        return _with_retries(
            lambda: _openai_compatible_generate(
                config,
                prompt,
                system_instruction=system_instruction,
                json_mode=json_mode,
                temperature=temperature,
            )
        )
    raise ValueError(f"unknown LLM adapter: {config.adapter_name}")
