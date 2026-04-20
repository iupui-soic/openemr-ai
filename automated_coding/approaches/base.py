"""Predictor interface that every benchmark approach implements."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Predictor(Protocol):
    name: str

    def prepare(
        self, label_space: list[str], descriptions: dict[str, str]
    ) -> None:
        """One-time setup: load models, pre-embed descriptions, compile regex, etc."""
        ...

    def predict(self, text: str) -> set[str]:
        """Return the predicted set of CPT codes for a single note."""
        ...

    def close(self) -> None:
        """Release any heavy resources (GPU memory, API clients)."""
        ...
