from pathlib import Path

import numpy as np
import pandas as pd

from emg_normalisation.training import (
    BUILT_IN_MODELS,
    built_in_model_factory,
    compare_linear_bundles,
    evaluate_model_factory_loso,
    fit_final_linear_bundle,
    manuscript_training_rows,
    rebuild_published_models,
    resolve_model_factory,
)


def feature_table() -> pd.DataFrame:
    rows = []
    centre = 0
    for participant_index in range(1, 8):
        for muscle in ("vaslat", "soleus"):
            for window in range(3):
                variance = participant_index * 0.01 + window * 0.001
                width = participant_index * 0.02 + window * 0.002
                rows.append(
                    {
                        "participant": f"C{participant_index:02d}",
                        "trial": "Cycling01",
                        "channel": f"r_{muscle}",
                        "model_muscle": muscle,
                        "centre_index": centre,
                        "variance_mV2": variance,
                        "kde_fwhm_mV": width,
                        "reference_normalised_target": 0.1 + 0.5 * variance + 0.25 * width,
                        "include_training": True,
                        "qc_status": "included",
                    }
                )
                centre += 1
    rows.extend(
        [
            {**rows[0], "participant": "E01", "reference_normalised_target": 100.0},
            {**rows[1], "include_training": False, "reference_normalised_target": 100.0},
            {**rows[2], "qc_status": "excluded", "reference_normalised_target": 100.0},
        ]
    )
    return pd.DataFrame(rows)


def test_manuscript_rows_enforce_participants_roles_and_qc():
    filtered = manuscript_training_rows(feature_table())
    assert set(filtered["participant"]) == {f"C{index:02d}" for index in range(1, 8)}
    assert filtered["include_training"].all()
    assert filtered["qc_status"].eq("included").all()
    assert len(filtered) == 42


def test_final_bundle_recovers_linear_relationship():
    table = feature_table()
    bundle = fit_final_linear_bundle(table)
    assert set(bundle.models) == {"soleus", "vaslat"}
    model = bundle.models["vaslat"]
    points = np.array([[0.03, 0.06], [0.07, 0.14]])
    expected = 0.1 + 0.5 * points[:, 0] + 0.25 * points[:, 1]
    assert np.allclose(model.predict(points), expected)


def test_built_in_models_share_fit_predict_contract():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(50, 2))
    targets = 0.2 + 0.1 * features[:, 0] - 0.05 * features[:, 1]
    for name in BUILT_IN_MODELS:
        model = built_in_model_factory(name)()
        predictions = model.fit(features, targets).predict(features[:3])
        assert predictions.shape == (3,)
        assert np.isfinite(predictions).all()


def test_unknown_model_name_has_actionable_error():
    try:
        resolve_model_factory("not-a-model")
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("unknown model should fail")
    assert "hist-gradient-boosting" in message
    assert "path/to/module.py:function" in message


def test_built_in_alternative_runs_through_shared_loso(tmp_path: Path):
    predictions, metrics = evaluate_model_factory_loso(
        feature_table(),
        built_in_model_factory("ridge"),
        tmp_path / "ridge_folds",
    )
    assert len(predictions) == 42
    assert len(metrics) == 14
    assert len(list((tmp_path / "ridge_folds").glob("*.model"))) == 14


def test_rebuild_exports_loso_and_final_artifacts(tmp_path: Path):
    paths = rebuild_published_models(feature_table(), tmp_path)
    assert set(paths) == {"model", "coefficients", "pickle", "predictions", "metrics", "summary"}
    assert all(path.exists() for path in paths.values())
    assert len(pd.read_csv(paths["metrics"])) == 14
    rebuilt = fit_final_linear_bundle(feature_table())
    comparison = compare_linear_bundles(rebuilt, rebuilt)
    assert comparison["absolute_difference"].max() == 0.0
