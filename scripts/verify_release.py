"""Run release-level integrity checks without repeating full model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from emg_normalisation.config import CHANNEL_EXCLUSIONS, MODEL_MUSCLES
from emg_normalisation.models import PublishedModelBundle
from emg_normalisation.training import compare_linear_bundles, fit_final_linear_bundle

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    trial_manifest = pd.read_csv(ROOT / "data/manifests/trial_manifest.csv")
    assert len(trial_manifest) == 246
    assert set(trial_manifest["participant"]) == {
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "E01",
    }
    model = PublishedModelBundle.load(ROOT / "models/cycling_linear/model.json")
    assert set(model.models) == set(MODEL_MUSCLES)
    maxima = pd.read_csv(ROOT / "data/manifests/reference_maxima.csv")
    for participant, channels in CHANNEL_EXCLUSIONS.items():
        assert not (
            maxima["participant"].eq(participant) & maxima["channel"].isin(channels)
        ).any()
    roles = pd.read_csv(ROOT / "data/manifests/trial_roles.csv")
    assert not roles.loc[roles["participant"].eq("E01"), "include_training"].any()
    references = pd.read_csv(ROOT / "data/manifests/reference_envelope_manifest.csv")
    assert len(references) == 246
    features_path = ROOT / "data/derived/window_features.parquet"
    if features_path.exists():
        features = pd.read_parquet(
            features_path,
            columns=[
                "participant",
                "trial",
                "channel",
                "model_muscle",
                "centre_index",
                "variance_mV2",
                "kde_fwhm_mV",
                "reference_normalised_target",
                "include_training",
                "qc_status",
            ],
        )
        assert {f"C{i:02d}" for i in range(1, 8)}.issubset(set(features["participant"]))
        for participant, channels in CHANNEL_EXCLUSIONS.items():
            assert not (
                features["participant"].eq(participant) & features["channel"].isin(channels)
            ).any()
        rebuilt = fit_final_linear_bundle(features)
        differences = compare_linear_bundles(rebuilt, model)
        assert differences["absolute_difference"].max() < 1e-10
    example = ROOT / "examples/full_trial_data/C06/trial_ffc10db23f/emgBPF.mot"
    assert example.exists() and example.stat().st_size > 1_000_000
    print("Release integrity checks passed.")


if __name__ == "__main__":
    main()
