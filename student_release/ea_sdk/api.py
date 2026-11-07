from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, artifacts_dir):
        """Load packaged assets."""

    @abstractmethod
    def reset(self, seed):
        """Reset mutable inference state."""

    @abstractmethod
    def update(self, window, now) -> int:
        """Predict from observations through the current row, never label."""
