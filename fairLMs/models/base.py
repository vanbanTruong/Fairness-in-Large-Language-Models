"""Model adapter base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class LoadedModel:
    """Tokenizer + model + device bundle returned by loaders."""

    name: str
    tokenizer: Any
    model: Any
    device: torch.device
    task: str
    profile: Any = None

    def to(self, device: torch.device) -> "LoadedModel":
        self.model.to(device)
        self.device = device
        return self

    def eval(self) -> "LoadedModel":
        self.model.eval()
        return self


class ModelAdapter(ABC):
    """Thin wrapper around a concrete model backend."""

    name: str

    @abstractmethod
    def load(self) -> LoadedModel:
        """Load tokenizer/model onto the configured device."""
