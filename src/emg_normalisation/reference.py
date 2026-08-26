"""Build conventional reference-normalised envelopes from bandpass EMG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import MANUSCRIPT_AMPLIFIER_GAIN, removeGain
from .config import REFERENCE_OVERRIDES, is_channel_excluded
from .io import iter_trial_files, read_emg_mot, write_parquet_or_pickle
from .preprocessing import conventional_linear_envelope, reference_normalise


def calculate_reference_maxima(
    bandpass_root: str | Path,
    reference_overrides: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Calculate audited participant-muscle maxima from all available trials.

    Parameters
    ----------
    bandpass_root:
        Directory organised as ``participant/trial/emgBPF.mot``.
    reference_overrides:
        Optional participant/channel-to-trial mapping. Defaults to the final
        manuscript overrides.

    Returns
    -------
    pandas.DataFrame
        One row per participant and side-specific channel, with maximum in mV,
        source trial, time, and selection mode.
    """

    overrides = reference_overrides if reference_overrides is not None else REFERENCE_OVERRIDES
    records: dict[tuple[str, str], dict[str, object]] = {}
    override_trials = {
        (participant, trial)
        for participant, channel_trials in overrides.items()
        for trial in channel_trials.values()
    }
    trial_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for participant, trial, path in iter_trial_files(bandpass_root):
        calibrated = removeGain(
            read_emg_mot(path),
            amplifier_gain=MANUSCRIPT_AMPLIFIER_GAIN,
        )
        envelope = conventional_linear_envelope(calibrated)
        if (participant, trial) in override_trials:
            trial_cache[(participant, trial)] = envelope
        time = envelope["time"].to_numpy(dtype=float)
        for channel in envelope.columns:
            if channel == "time" or is_channel_excluded(participant, channel):
                continue
            values = envelope[channel].to_numpy(dtype=float)
            if not np.isfinite(values).any():
                continue
            index = int(np.nanargmax(values))
            key = (participant, channel)
            if key not in records or float(values[index]) > float(records[key]["maximum_mV"]):
                records[key] = {
                    "participant": participant,
                    "channel": channel,
                    "reference_trial": trial,
                    "maximum_mV": float(values[index]),
                    "time_at_max_s": float(time[index]),
                    "reference_selection": "participant_dynamic_maximum",
                }
    for participant, mapping in overrides.items():
        for channel, trial in mapping.items():
            envelope = trial_cache.get((participant, trial))
            if envelope is None or channel not in envelope:
                raise KeyError(f"Reference override unavailable: {participant}/{trial}/{channel}")
            values = envelope[channel].to_numpy(dtype=float)
            index = int(np.nanargmax(values))
            records[(participant, channel)] = {
                "participant": participant,
                "channel": channel,
                "reference_trial": trial,
                "maximum_mV": float(values[index]),
                "time_at_max_s": float(envelope["time"].iloc[index]),
                "reference_selection": "protocol_or_qc_override",
            }
    return pd.DataFrame(records.values()).sort_values(["participant", "channel"]).reset_index(drop=True)


def build_reference_envelopes(
    bandpass_root: str | Path,
    output_root: str | Path,
    maxima: pd.DataFrame,
) -> pd.DataFrame:
    """Generate compressed reference-normalised envelopes for every trial."""

    required = {"participant", "channel", "maximum_mV"}
    if not required.issubset(maxima.columns):
        raise ValueError(f"maxima must include {sorted(required)}")
    maxima_by_participant = {
        participant: dict(zip(group["channel"], group["maximum_mV"], strict=True))
        for participant, group in maxima.groupby("participant", sort=False)
    }
    output_dir = Path(output_root)
    rows: list[dict[str, object]] = []
    for participant, trial, path in iter_trial_files(bandpass_root):
        calibrated = removeGain(
            read_emg_mot(path),
            amplifier_gain=MANUSCRIPT_AMPLIFIER_GAIN,
        )
        envelope = conventional_linear_envelope(calibrated)
        excluded = [channel for channel in envelope.columns if is_channel_excluded(participant, channel)]
        envelope = envelope.drop(columns=excluded, errors="ignore")
        reference = reference_normalise(envelope, maxima_by_participant.get(participant, {}))
        destination = output_dir / participant / trial / "reference_normalised_envelope.parquet"
        written = write_parquet_or_pickle(reference, destination)
        rows.append(
            {
                "participant": participant,
                "trial": trial,
                "path": written.relative_to(output_dir).as_posix(),
                "samples": len(reference),
                "channels": len(reference.columns) - 1,
            }
        )
    return pd.DataFrame(rows).sort_values(["participant", "trial"]).reset_index(drop=True)
