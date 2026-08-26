"""Build reusable window-level modelling tables from signal-level data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import MANUSCRIPT_AMPLIFIER_GAIN, removeGain
from .config import anatomical_muscle, is_channel_excluded
from .features import extract_window_features
from .io import iter_trial_files, read_emg_mot, read_table, write_parquet_or_pickle
from .preprocessing import infer_sampling_rate

FEATURE_COLUMNS = ("variance_mV2", "kde_fwhm_mV")


def extract_feature_dataset(
    data_root: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Extract the complete QC-filtered window-feature dataset.

    Parameters
    ----------
    data_root:
        Repository ``data`` directory containing ``bandpass_emg``,
        ``reference_envelopes`` and ``manifests/trial_roles.csv``.
    output_path:
        Destination Parquet path.
    overwrite:
        Replace an existing output when true.

    Returns
    -------
    pathlib.Path
        Path to the compressed feature table.

    Notes
    -----
    Variance is returned in mV squared and KDE FWHM in mV. The target is the
    conventional reference-normalised envelope value at each window centre.

    Examples
    --------
    >>> extract_feature_dataset("data", "outputs/window_features.parquet")
    PosixPath('outputs/window_features.parquet')
    """

    root = Path(data_root)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass overwrite=True to replace it")
    role_path = root / "manifests" / "trial_roles.csv"
    roles = pd.read_csv(role_path).set_index(["participant", "trial"]) if role_path.exists() else None
    rows: list[pd.DataFrame] = []
    for participant, trial, emg_path in iter_trial_files(root / "bandpass_emg"):
        reference_path = (
            root
            / "reference_envelopes"
            / participant
            / trial
            / "reference_normalised_envelope.parquet"
        )
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Missing reference envelope for {participant}/{trial}; run build-reference first"
            )
        calibrated = removeGain(
            read_emg_mot(emg_path),
            amplifier_gain=MANUSCRIPT_AMPLIFIER_GAIN,
        )
        reference = read_table(reference_path)
        sampling_rate = infer_sampling_rate(calibrated["time"].to_numpy(dtype=float))
        role = roles.loc[(participant, trial)] if roles is not None else None
        for channel in calibrated.columns:
            if (
                channel == "time"
                or channel not in reference
                or is_channel_excluded(participant, channel)
            ):
                continue
            features = extract_window_features(
                calibrated[channel].to_numpy(dtype=float),
                sampling_rate,
            )
            if not features.centre_indices.size:
                continue
            centres = features.centre_indices
            targets = reference[channel].to_numpy(dtype=float)[centres]
            finite = (
                np.isfinite(features.variance_mV2)
                & np.isfinite(features.kde_fwhm_mV)
                & np.isfinite(targets)
            )
            rows.append(
                pd.DataFrame(
                    {
                        "participant": participant,
                        "trial": trial,
                        "channel": channel,
                        "model_muscle": anatomical_muscle(channel),
                        "centre_index": centres[finite],
                        "centre_time_s": calibrated["time"].to_numpy(dtype=float)[centres[finite]],
                        "variance_mV2": features.variance_mV2[finite],
                        "kde_fwhm_mV": features.kde_fwhm_mV[finite],
                        "reference_normalised_target": targets[finite],
                        "loso_fold": participant,
                        "include_training": bool(role["include_training"]) if role is not None else True,
                        "include_primary_reporting": (
                            bool(role["include_primary_reporting"]) if role is not None else True
                        ),
                        "qc_status": "included",
                    }
                )
            )
    if not rows:
        raise ValueError("No valid feature rows were extracted")
    return write_parquet_or_pickle(pd.concat(rows, ignore_index=True), destination)


def load_training_features(
    path: str | Path,
    *,
    cycling_only: bool = True,
) -> pd.DataFrame:
    """Load valid feature rows for model development.

    Parameters
    ----------
    path:
        Feature Parquet or CSV.
    cycling_only:
        Restrict training to the seven cycling participants when true.

    Returns
    -------
    pandas.DataFrame
        QC-filtered modelling rows with public participant labels.
    """

    frame = read_table(path)
    required = {
        "participant",
        "model_muscle",
        "variance_mV2",
        "kde_fwhm_mV",
        "reference_normalised_target",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    if "qc_status" in frame:
        frame = frame.loc[frame["qc_status"].eq("included")]
    if "include_training" in frame:
        frame = frame.loc[frame["include_training"].astype(bool)]
    if cycling_only:
        frame = frame.loc[frame["participant"].str.fullmatch(r"P[1-7]")]
    return frame.reset_index(drop=True)
