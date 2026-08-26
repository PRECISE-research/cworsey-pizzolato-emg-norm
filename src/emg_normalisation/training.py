"""Fixed LOSO splits and extension points for alternative model classes."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterator
from importlib import util
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CYCLING_PARTICIPANTS, DEFAULT_WINDOW_CONFIG, MUSCLE_LABELS
from .models import (
    EMGAmplitudeModel,
    PublishedModelBundle,
    SklearnEstimatorAdapter,
    StandardisedLinearModel,
)

FEATURE_COLUMNS = ("variance_mV2", "kde_fwhm_mV")
TARGET_COLUMN = "reference_normalised_target"
BUILT_IN_MODELS = {
    "linear": "Published ordinary least-squares baseline",
    "ridge": "L2-regularised linear regression with fold-specific feature scaling",
    "hist-gradient-boosting": "Scalable nonlinear histogram gradient boosting",
    "random-forest": "Nonlinear random-forest ensemble with deterministic defaults",
}


def manuscript_training_rows(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Return the QC-approved C01-C07 rows used for model development."""

    data = feature_table.copy()
    if "qc_status" in data.columns:
        data = data[data["qc_status"].eq("included")]
    if "include_training" in data.columns:
        data = data[data["include_training"].astype(bool)]
    elif "include" in data.columns:
        data = data[data["include"].astype(bool)]
    data = data[data["participant"].astype(str).isin(CYCLING_PARTICIPANTS)]
    return data.reset_index(drop=True)


def loso_splits(
    frame: pd.DataFrame,
    participant_column: str = "participant",
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield deterministic leave-one-participant-out masks."""

    participants = sorted(frame[participant_column].dropna().astype(str).unique())
    for held_out in participants:
        test = frame[participant_column].astype(str).eq(held_out).to_numpy()
        yield held_out, ~test, test


def load_model_factory(specification: str) -> Callable[[], EMGAmplitudeModel]:
    """Load ``path/to/module.py:function`` as a zero-argument model factory."""

    try:
        module_path, function_name = specification.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("custom model must use path.py:function syntax") from exc
    path = Path(module_path).resolve()
    module_spec = util.spec_from_file_location("_emg_user_model", path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"{function_name!r} is not callable")
    return factory


def linear_model_factory(muscle: str | None = None) -> StandardisedLinearModel:
    """Create an unfitted manuscript-form linear model."""

    return StandardisedLinearModel(
        feature_names=FEATURE_COLUMNS,
        feature_mean=np.zeros(2),
        feature_sd=np.ones(2),
        intercept=0.0,
        coefficients=np.zeros(2),
        muscle=muscle,
    )


def built_in_model_factory(name: str) -> Callable[[], EMGAmplitudeModel]:
    """Return a fresh-estimator factory for a named built-in model.

    All stochastic estimators use a fixed seed. These defaults are intended to
    make first comparisons easy, not to claim that their hyperparameters are
    optimal for this dataset.
    """

    if name == "linear":
        return linear_model_factory
    if name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def create_ridge() -> EMGAmplitudeModel:
            return SklearnEstimatorAdapter(make_pipeline(StandardScaler(), Ridge(alpha=1.0)))

        return create_ridge
    if name == "hist-gradient-boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        def create_hist_gradient_boosting() -> EMGAmplitudeModel:
            return SklearnEstimatorAdapter(
                HistGradientBoostingRegressor(
                    learning_rate=0.08,
                    max_iter=100,
                    min_samples_leaf=20,
                    random_state=1,
                )
            )

        return create_hist_gradient_boosting
    if name == "random-forest":
        from sklearn.ensemble import RandomForestRegressor

        def create_random_forest() -> EMGAmplitudeModel:
            return SklearnEstimatorAdapter(
                RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=20,
                    max_samples=0.5,
                    random_state=1,
                    n_jobs=-1,
                )
            )

        return create_random_forest
    available = ", ".join(BUILT_IN_MODELS)
    raise ValueError(f"unknown built-in model {name!r}; choose one of: {available}")


def resolve_model_factory(specification: str) -> Callable[[], EMGAmplitudeModel]:
    """Resolve a built-in name or a ``path.py:function`` custom factory."""

    if specification in BUILT_IN_MODELS:
        return built_in_model_factory(specification)
    if ":" in specification:
        return load_model_factory(specification)
    available = ", ".join(BUILT_IN_MODELS)
    raise ValueError(
        f"unknown model {specification!r}; choose a built-in ({available}) "
        "or provide path/to/module.py:function"
    )


def evaluate_model_factory_loso(
    feature_table: pd.DataFrame,
    factory: Callable[[], EMGAmplitudeModel],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one model per anatomical muscle in each fixed LOSO fold.

    The function is intentionally model-agnostic. Any returned object with
    ``fit`` and ``predict`` can participate. Predictions are bounded to 0-1
    only after the model returns, matching the manuscript reconstruction.
    """

    required = {
        "participant",
        "trial",
        "channel",
        "model_muscle",
        "centre_index",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"feature table missing columns: {sorted(missing)}")
    data = manuscript_training_rows(feature_table)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for held_out, train_mask, test_mask in loso_splits(data):
        for muscle in sorted(data["model_muscle"].dropna().unique()):
            train = data.loc[train_mask & data["model_muscle"].eq(muscle)]
            test = data.loc[test_mask & data["model_muscle"].eq(muscle)]
            if train.empty or test.empty:
                continue
            model = factory()
            model.fit(train.loc[:, FEATURE_COLUMNS].to_numpy(), train[TARGET_COLUMN].to_numpy())
            predicted = np.clip(model.predict(test.loc[:, FEATURE_COLUMNS].to_numpy()), 0.0, 1.0)
            result = test[["participant", "trial", "channel", "model_muscle", "centre_index"]].copy()
            result["reference_normalised_target"] = test[TARGET_COLUMN].to_numpy()
            result["prediction"] = predicted
            result["held_out_participant"] = held_out
            prediction_parts.append(result)
            residual = result["reference_normalised_target"].to_numpy() - predicted
            total = np.sum(
                (result["reference_normalised_target"] - result["reference_normalised_target"].mean()) ** 2
            )
            metric_rows.append(
                {
                    "held_out_participant": held_out,
                    "model_muscle": muscle,
                    "n": len(result),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "r2": float(1 - np.sum(residual**2) / total) if total > 0 else np.nan,
                }
            )
            model.save(output / f"{held_out}_{muscle}.model")
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics


def fit_final_linear_bundle(feature_table: pd.DataFrame) -> PublishedModelBundle:
    """Fit the released model form to all QC-approved cycling participants."""

    required = {"participant", "model_muscle", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"feature table missing columns: {sorted(missing)}")
    data = manuscript_training_rows(feature_table)
    if data.empty:
        raise ValueError("no QC-approved C01-C07 training rows were found")
    models: dict[str, StandardisedLinearModel] = {}
    for muscle in sorted(data["model_muscle"].dropna().astype(str).unique()):
        rows = data[data["model_muscle"].astype(str).eq(muscle)]
        model = linear_model_factory(muscle)
        model.fit(rows.loc[:, FEATURE_COLUMNS].to_numpy(), rows[TARGET_COLUMN].to_numpy())
        models[muscle] = model
    return PublishedModelBundle(
        models=models,
        metadata={
            "name": "cycling-cohort side-pooled linear EMG amplitude models",
            "version": "1.0.0",
            "participants": list(CYCLING_PARTICIPANTS),
            "model_scope": "side_pooled_anatomical_muscle",
            "input_unit": "input_referred_mV",
            "feature_units": {"variance_mV2": "mV^2", "kde_fwhm_mV": "mV"},
            "target": "reference-normalised envelope at window centre",
            "output_bounds": [0.0, 1.0],
            "training_trials": "17 cycling conditions plus three cycling-cohort CMJ trials",
            "external_dataset_in_training": False,
            "post_processing": "trial peak scales conventional 6-Hz envelope",
            "prediction_floor_subtraction": False,
            "conventional_envelope_floor_subtraction": False,
        },
    )


def coefficient_table(bundle: PublishedModelBundle) -> pd.DataFrame:
    """Return one transparent coefficient row per anatomical muscle."""

    rows = []
    for muscle, model in sorted(bundle.models.items()):
        rows.append(
            {
                "model_muscle": muscle,
                "label": MUSCLE_LABELS.get(muscle, muscle),
                "beta_0": model.intercept,
                "beta_variance": model.coefficients[0],
                "beta_kde_fwhm": model.coefficients[1],
                "variance_mean_mV2": model.feature_mean[0],
                "variance_sd_mV2": model.feature_sd[0],
                "kde_fwhm_mean_mV": model.feature_mean[1],
                "kde_fwhm_sd_mV": model.feature_sd[1],
                "window_s": DEFAULT_WINDOW_CONFIG.window_s,
                "step_s": DEFAULT_WINDOW_CONFIG.step_s,
                "minimum_valid_s": DEFAULT_WINDOW_CONFIG.min_valid_s,
            }
        )
    return pd.DataFrame(rows)


def compare_linear_bundles(
    rebuilt: PublishedModelBundle,
    reference: PublishedModelBundle,
) -> pd.DataFrame:
    """Compare rebuilt and released model parameters without rounding."""

    rebuilt_table = coefficient_table(rebuilt).set_index("model_muscle").select_dtypes(include="number")
    reference_table = coefficient_table(reference).set_index("model_muscle").select_dtypes(include="number")
    rows = []
    for muscle in sorted(set(rebuilt_table.index) | set(reference_table.index)):
        for parameter in sorted(set(rebuilt_table.columns) | set(reference_table.columns)):
            rebuilt_value = rebuilt_table.at[muscle, parameter] if muscle in rebuilt_table.index else np.nan
            reference_value = reference_table.at[muscle, parameter] if muscle in reference_table.index else np.nan
            rows.append(
                {
                    "model_muscle": muscle,
                    "parameter": parameter,
                    "rebuilt": rebuilt_value,
                    "reference": reference_value,
                    "absolute_difference": abs(rebuilt_value - reference_value),
                }
            )
    return pd.DataFrame(rows)


def rebuild_published_models(
    feature_table: pd.DataFrame,
    output_dir: str | Path,
    *,
    reference_model: str | Path | None = None,
) -> dict[str, Path]:
    """Run manuscript LOSO validation and rebuild the final linear bundle."""

    output = Path(output_dir)
    folds = output / "loso_models"
    output.mkdir(parents=True, exist_ok=True)
    predictions, metrics = evaluate_model_factory_loso(feature_table, linear_model_factory, folds)
    predictions_path = output / "loso_window_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    metrics_path = output / "loso_window_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    bundle = fit_final_linear_bundle(feature_table)
    model_path = bundle.save(output / "model.json")
    coefficients_path = output / "coefficients.csv"
    coefficient_table(bundle).to_csv(coefficients_path, index=False)
    pickle_path = output / "model.pkl"
    with pickle_path.open("wb") as stream:
        pickle.dump(bundle, stream)

    paths = {
        "model": model_path,
        "coefficients": coefficients_path,
        "pickle": pickle_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
    }
    if reference_model is not None:
        reference = PublishedModelBundle.load(reference_model)
        comparison_path = output / "published_model_comparison.csv"
        compare_linear_bundles(bundle, reference).to_csv(comparison_path, index=False)
        paths["comparison"] = comparison_path
    summary_path = output / "rebuild_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "participants": list(CYCLING_PARTICIPANTS),
                "n_training_rows": len(manuscript_training_rows(feature_table)),
                "n_loso_predictions": len(predictions),
                "n_models": len(bundle.models),
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["summary"] = summary_path
    return paths
