"""Apply the published model to a user's bandpass-filtered EMG trial.

Examples
--------
Amplified EMG stored in volts with a known amplifier gain of 1000:

python examples/predict_own_emg.py my_bandpass.mot outputs/my_estimate.csv \
    --amplifier-gain 1000

EMG already expressed as input-referred mV:

python examples/predict_own_emg.py my_bandpass.csv outputs/my_estimate.csv \
    --already-input-mv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from emg_normalisation import load_published_model, predict_trial, read_emg_mot
from emg_normalisation.calibration import removeGain
from emg_normalisation.io import read_table

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "cycling_linear" / "model.json"


def load_bandpass(path: Path) -> pd.DataFrame:
    """Read OpenSim MOT, CSV, CSV.GZ, or Parquet bandpass EMG."""

    return read_emg_mot(path) if path.suffix.lower() == ".mot" else read_table(path)


def build_parser() -> argparse.ArgumentParser:
    """Build the user-data example command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Estimate normalised EMG envelopes from one bandpass-filtered trial "
            "using the published cycling-cohort model."
        )
    )
    parser.add_argument("input", type=Path, help="Bandpass EMG MOT, CSV, or Parquet file.")
    parser.add_argument("output", type=Path, help="Destination CSV for estimated envelopes.")
    units = parser.add_mutually_exclusive_group(required=True)
    units.add_argument(
        "--amplifier-gain",
        type=float,
        help="Known voltage gain applied to the stored EMG.",
    )
    units.add_argument(
        "--already-input-mv",
        action="store_true",
        help="Confirm that the input is already gain-removed, input-referred mV.",
    )
    parser.add_argument(
        "--stored-unit",
        choices=("V", "mV", "uV"),
        default="V",
        help="Voltage unit before gain removal; default: V.",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        help="Input time-column name; values must be seconds.",
    )
    parser.add_argument(
        "--channel-map",
        type=Path,
        help="Optional JSON mapping from input channel names to published names.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("scaled_envelope", "model_trace"),
        default="scaled_envelope",
        help="Final trial-scaled envelope or direct interpolated model trace.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load user EMG, apply the published model, and save prediction metadata."""

    args = build_parser().parse_args(argv)
    frame = load_bandpass(args.input)
    if args.time_column != "time":
        if args.time_column not in frame:
            raise ValueError(f"Time column {args.time_column!r} was not found")
        frame = frame.rename(columns={args.time_column: "time"})
    if "time" not in frame:
        raise ValueError("Input must contain time in seconds")

    channel_map: dict[str, str] = {}
    if args.channel_map:
        channel_map = json.loads(args.channel_map.read_text(encoding="utf-8"))
        frame = frame.rename(columns=channel_map)
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Channel mapping created duplicate columns: {duplicates}")

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric["time"]).all():
        raise ValueError("Time contains missing or non-numeric values")
    if not numeric["time"].is_monotonic_increasing:
        raise ValueError("Time must increase monotonically")

    if args.already_input_mv:
        input_mV = numeric
        gain_metadata: dict[str, object] = {
            "input_unit": "input_referred_mV",
            "amplifier_gain_removed_by_script": False,
        }
    else:
        input_mV = removeGain(
            numeric,
            amplifier_gain=args.amplifier_gain,
            stored_unit=args.stored_unit,
        )
        gain_metadata = {
            "stored_unit": args.stored_unit,
            "amplifier_gain": args.amplifier_gain,
            "output_unit": "input_referred_mV",
            "amplifier_gain_removed_by_script": True,
        }

    model = load_published_model(args.model)
    result = predict_trial(input_mV, model, output_mode=args.output_mode)
    predicted_channels = [column for column in result.output if column != "time"]
    skipped_channels = [
        column
        for column in input_mV
        if column != "time" and column not in predicted_channels
    ]
    if not predicted_channels:
        raise ValueError(
            "No recognised muscle channels were found. Use --channel-map and "
            "examples/channel_map_template.json."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.output.to_csv(args.output, index=False, float_format="%.12g")
    metadata = {
        "input": str(args.input.resolve()),
        "model": str(args.model.resolve()),
        "output_mode": args.output_mode,
        "time_unit": "s",
        "output_unit": "normalised_excitation_0_to_1",
        "predicted_channels": predicted_channels,
        "skipped_channels": skipped_channels,
        "estimated_peak_by_channel": result.estimated_peak_by_channel,
        "channel_map": channel_map,
        "gain_handling": gain_metadata,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved: {args.output}")
    print(f"Metadata: {metadata_path}")
    print(f"Predicted channels ({len(predicted_channels)}): {', '.join(predicted_channels)}")
    if skipped_channels:
        print(f"Skipped unrecognised channels: {', '.join(skipped_channels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
