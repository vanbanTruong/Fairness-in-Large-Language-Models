"""OpenAI API client helpers used by several decoder-only metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from fairLMs.models.base import ModelAdapter


@dataclass
class OpenAILoadedModel:
    """Lightweight stand-in for API-backed models (no local weights)."""

    name: str
    client: Any
    model: str
    task: str = "openai"


class OpenAIModel(ModelAdapter):
    """Create an OpenAI client + default completion model name.

    Reads ``OPENAI_API_KEY`` from the environment when ``api_key`` is omitted.
    """

    #: Mirrors :attr:`OpenAILoadedModel.task` so applicability can be decided
    #: before the client is constructed. An API-served decoder exposes text and
    #: nothing else, which is what makes it refuse activation-based components.
    #: See :data:`fairLMs.applicability.TASK_PROFILES`.
    task: str = "openai"

    def __init__(
        self,
        model_name: str = "davinci-002",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.name = model_name
        self.api_key = api_key
        self._loaded: Optional[OpenAILoadedModel] = None

    def load(self) -> OpenAILoadedModel:
        if self._loaded is not None:
            return self._loaded

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIModel. Install with: pip install openai"
            ) from exc

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=key) if key else OpenAI()
        self._loaded = OpenAILoadedModel(
            name=self.model_name,
            client=client,
            model=self.model_name,
        )
        return self._loaded


def get_openai_client(api_key: Optional[str] = None):
    """Return an OpenAI client (shared helper for leaf scripts)."""
    return OpenAIModel(api_key=api_key).load().client
