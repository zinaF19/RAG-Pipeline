"""Shared native chat-model setup; retrieval and agent logic stay in the notebooks."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import SecretStr

load_dotenv(Path(__file__).resolve().parent / ".env")

PROVIDERS = {
    "groq": ("groq", "openai/gpt-oss-20b"),
    "mistral": ("mistralai", "ministral-14b-latest"),
    "openrouter": ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    "google": ("google_genai", "gemma-4-26b-a4b-it"),
}


def create_chat_model(
    *, temperature: float = 0, max_tokens: int = 1024
) -> BaseChatModel:
    """Select only the requested provider and its own key, without API calls."""
    name = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of: {', '.join(PROVIDERS)}")
    key = os.getenv(f"{name.upper()}_API_KEY", "").strip()
    if not key:
        raise ValueError(f"Set {name.upper()}_API_KEY in .env for LLM_PROVIDER={name}")
    adapter, default = PROVIDERS[name]
    model = os.getenv(f"{name.upper()}_MODEL", default)
    options: dict[str, Any] = {}
    if name == "openrouter":
        fallback = os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/free").strip()
        options = {
            "model_kwargs": {
                "retries": None,
                "models": [model, fallback] if fallback else None,
            },
            "openrouter_provider": {"require_parameters": True},
        }
    elif name == "google":
        options = {"vertexai": False}
    return init_chat_model(
        model=model,
        model_provider=adapter,
        api_key=SecretStr(key),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30000 if name == "openrouter" else 30,
        max_retries=0,
        **options,
    )


def validate_response(message: BaseMessage) -> BaseMessage:
    """Reject incomplete or empty final answers before parsing away metadata."""
    reason = message.response_metadata.get("finish_reason")
    if str(reason).lower() != "stop":
        raise ValueError(f"Incomplete provider response: {reason}")
    if not message.text.strip():
        raise ValueError("The provider returned no answer text")
    return message


def response_text(message: BaseMessage) -> str:
    """Return only visible answer text, never reasoning or other content blocks."""
    return validate_response(message).text
