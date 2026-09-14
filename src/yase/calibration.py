"""Lightweight confidence calibration without a scientific Python dependency."""

from collections.abc import Iterable

import numpy as np


class TemperatureScaler:
    """Calibrate scores in ``[0, 1]`` with one learned temperature.

    This is temperature scaling on logits, trained with binary cross entropy.
    It is intentionally small enough for edge deployments and should be fit on
    held-out validation data, never on the test set.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def transform(self, score: float) -> float:
        value = float(np.clip(score, 1e-6, 1 - 1e-6))
        logit = np.log(value / (1.0 - value))
        calibrated = 1.0 / (1.0 + np.exp(-logit / self.temperature))
        return float(np.clip(calibrated, 0.0, 1.0))

    def transform_many(self, scores: Iterable[float]) -> np.ndarray:
        return np.asarray([self.transform(score) for score in scores], dtype=np.float32)

    def fit(
        self,
        scores: Iterable[float],
        labels: Iterable[float],
        steps: int = 200,
        learning_rate: float = 0.05,
    ) -> "TemperatureScaler":
        """Fit temperature by gradient descent on held-out binary labels."""
        if steps < 1 or learning_rate <= 0:
            raise ValueError(
                "steps must be positive and learning_rate must be positive"
            )
        probabilities = np.clip(
            np.asarray(list(scores), dtype=np.float64), 1e-6, 1 - 1e-6
        )
        targets = np.asarray(list(labels), dtype=np.float64)
        if probabilities.shape != targets.shape or probabilities.size == 0:
            raise ValueError(
                "scores and labels must be non-empty and have equal length"
            )
        if np.any((targets < 0) | (targets > 1)):
            raise ValueError("labels must be in [0, 1]")
        logits = np.log(probabilities / (1.0 - probabilities))
        log_temperature = float(np.log(self.temperature))
        for _ in range(steps):
            temperature = float(np.exp(log_temperature))
            scaled = logits / temperature
            predictions = 1.0 / (1.0 + np.exp(-np.clip(scaled, -50, 50)))
            gradient = np.mean((predictions - targets) * (-scaled))
            log_temperature -= learning_rate * float(gradient)
            log_temperature = float(
                np.clip(log_temperature, np.log(0.05), np.log(20.0))
            )
        self.temperature = float(np.exp(log_temperature))
        return self

    def to_dict(self) -> dict[str, float]:
        return {"temperature": self.temperature}


__all__ = ["TemperatureScaler"]
