"""Published linear models and adapters for user-defined estimators."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np


@runtime_checkable
class EMGAmplitudeModel(Protocol):
    """Minimal interface required by the training and evaluation pipeline."""

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Self:
        """Fit a model from ``(n_windows, n_features)`` inputs."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict one normalised amplitude per feature row."""

    def save(self, path: str | Path) -> Path:
        """Serialise the fitted model."""

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously fitted model."""


@dataclass
class StandardisedLinearModel:
    """Multiple least-squares model using training-fold feature scaling."""

    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_sd: np.ndarray
    intercept: float
    coefficients: np.ndarray
    muscle: str | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Self:
        """Fit standardisation parameters and ordinary least squares."""

        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be 2D and align with targets")
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if valid.sum() <= x.shape[1]:
            raise ValueError("insufficient finite rows to fit linear model")
        x = x[valid]
        y = y[valid]
        self.feature_mean = np.mean(x, axis=0)
        self.feature_sd = np.std(x, axis=0, ddof=0)
        self.feature_sd = np.where(self.feature_sd > 0, self.feature_sd, 1.0)
        design = np.column_stack((np.ones(len(x)), (x - self.feature_mean) / self.feature_sd))
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.intercept = float(beta[0])
        self.coefficients = np.asarray(beta[1:], dtype=float)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict unconstrained normalised amplitude."""

        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != len(self.coefficients):
            raise ValueError("feature column count does not match model")
        z = (x - self.feature_mean) / self.feature_sd
        return self.intercept + z @ self.coefficients

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable model definition."""

        return {
            "type": "standardised_linear_least_squares",
            "muscle": self.muscle,
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_sd": self.feature_sd.tolist(),
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Construct a model from its transparent JSON definition."""

        return cls(
            feature_names=tuple(payload["feature_names"]),
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_sd=np.asarray(payload["feature_sd"], dtype=float),
            intercept=float(payload["intercept"]),
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            muscle=payload.get("muscle"),
        )

    def save(self, path: str | Path) -> Path:
        """Write model JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load model JSON."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class SklearnEstimatorAdapter:
    """Wrap any scikit-learn-style estimator exposing ``fit`` and ``predict``."""

    def __init__(self, estimator: Any):
        if not callable(getattr(estimator, "fit", None)) or not callable(getattr(estimator, "predict", None)):
            raise TypeError("estimator must expose fit and predict")
        self.estimator = estimator

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Self:
        self.estimator.fit(np.asarray(features), np.asarray(targets).reshape(-1))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(np.asarray(features)), dtype=float).reshape(-1)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            pickle.dump(self.estimator, stream)
        return output

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with Path(path).open("rb") as stream:
            return cls(pickle.load(stream))


@dataclass
class PublishedModelBundle:
    """Collection of one side-pooled model per anatomical muscle."""

    models: dict[str, StandardisedLinearModel]
    metadata: dict[str, Any]

    def model_for_channel(self, channel: str) -> StandardisedLinearModel:
        """Resolve a side-specific channel to its anatomical-muscle model."""

        key = channel[2:] if channel.startswith(("l_", "r_")) else channel
        if key not in self.models:
            raise KeyError(f"No published model for channel {channel!r}")
        return self.models[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "models": {key: model.to_dict() for key, model in sorted(self.models.items())},
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> Self:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            models={key: StandardisedLinearModel.from_dict(item) for key, item in payload["models"].items()},
            metadata=dict(payload.get("metadata", {})),
        )


def load_published_model(path: str | Path | None = None) -> PublishedModelBundle:
    """Load the final cycling-cohort model bundle.

    When ``path`` is omitted, the function resolves the repository model at
    ``models/cycling_linear/model.json`` relative to the checkout.
    """

    if path is not None:
        return PublishedModelBundle.load(path)
    candidate = Path(__file__).resolve().parents[2] / "models" / "cycling_linear" / "model.json"
    if not candidate.exists():
        raise FileNotFoundError(
            "Published model not found. Pass an explicit model.json path or run from a repository checkout."
        )
    return PublishedModelBundle.load(candidate)
