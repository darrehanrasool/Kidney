"""Early stopping tracker to prevent overfitting."""

from typing import Dict, Any


class EarlyStopping:
    """Monitors validation metrics to stop training when progress plateaus."""

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, current_score: float) -> bool:
        """Update tracker with current epoch score.

        Args:
            current_score: Monitored metric value.

        Returns:
            True if early stopping condition met, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else: # min
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def state_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.patience = state.get("patience", self.patience)
        self.mode = state.get("mode", self.mode)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.counter = state.get("counter", 0)
        self.best_score = state.get("best_score")
        self.early_stop = state.get("early_stop", False)
