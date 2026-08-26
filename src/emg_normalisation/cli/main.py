"""Command-line interface for data, prediction, training, and reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..calibration import removeGain
from ..dataset import extract_feature_dataset
from ..evaluation import window_metrics
from ..io import read_emg_mot, read_table, write_parquet_or_pickle
from ..models import load_published_model
from ..prediction import predict_trial
from ..reference import build_reference_envelopes, calculate_reference_maxima
from ..training import (
    BUILT_IN_MODELS,
    evaluate_model_factory_loso,
    rebuild_published_models,
    resolve_model_factory,
)


def _validate_data(args: argparse.Namespace) -> int:
    """Validate the public data files supplied by this repository."""
    root = Path(args.data_root)
    manifest = root / "manifests" / "trial_manifest.csv"
    cache = root / "derived" / "window_features.parquet"
    example_root = Path(args.example_root)
    example_emg = example_root / "emgBPF.mot"
    example_reference = example_root / "reference_normalised_envelope.parquet"
    required_paths = (manifest, cache, example_emg, example_reference)
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        print("Missing required public data files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print("Run git lfs pull to download the bundled data files.", file=sys.stderr)
        return 1

    lfs_pointers = []
    for path in (cache, example_emg, example_reference):
        with path.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                lfs_pointers.append(path)
    if lfs_pointers:
        print("Git LFS files have not been downloaded:", file=sys.stderr)
        for path in lfs_pointers:
            print(f"  {path}", file=sys.stderr)
        print("Run git lfs pull to download the bundled data files.", file=sys.stderr)
        return 1

    columns = set(pq.ParquetFile(cache).schema.names)
    required_columns = {
        "participant",
        "trial",
        "channel",
        "variance_mV2",
        "kde_fwhm_mV",
        "reference_normalised_target",
    }
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        print(f"Feature cache is missing columns: {missing_columns}", file=sys.stderr)
        return 1
    example = read_emg_mot(example_emg)
    if len(example.columns) < 2:
        print("Example EMG file contains no signal channels.", file=sys.stderr)
        return 1

    print(f"Manifest rows: {len(pd.read_csv(manifest))}")
    print(f"C06 example EMG channels: {len(example.columns) - 1}")
    print(f"Feature-cache columns: {len(columns)}")
    print("Data validation passed.")
    return 0


def _build_reference(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    bandpass = data_root / "bandpass_emg"
    maxima = calculate_reference_maxima(bandpass)
    maxima_path = data_root / "manifests" / "reference_maxima.csv"
    maxima_path.parent.mkdir(parents=True, exist_ok=True)
    maxima.to_csv(maxima_path, index=False)
    manifest = build_reference_envelopes(bandpass, data_root / "reference_envelopes", maxima)
    manifest.to_csv(data_root / "manifests" / "reference_envelope_manifest.csv", index=False)
    print(f"Wrote {len(manifest)} reference envelope files.")
    return 0


def _predict(args: argparse.Namespace) -> int:
    frame = read_emg_mot(args.input)
    if args.input_unit != "mV":
        if args.gain is None:
            raise ValueError("--gain is required for amplified stored input")
        stored_units = {
            "stored_v": "V",
            "stored_mV": "mV",
            "stored_uV": "uV",
        }
        frame = removeGain(
            frame,
            amplifier_gain=args.gain,
            stored_unit=stored_units[args.input_unit],
        )
    model = load_published_model(args.model)
    result = predict_trial(frame, model, output_mode=args.output_mode)
    output = write_parquet_or_pickle(result.output, args.output)
    Path(str(output) + ".metadata.json").write_text(
        json.dumps(
            {
                "output_mode": result.output_mode,
                "estimated_peak_by_channel": result.estimated_peak_by_channel,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


def _train(args: argparse.Namespace) -> int:
    table = read_table(args.features)
    factory = resolve_model_factory(args.model)
    predictions, metrics = evaluate_model_factory_loso(table, factory, args.output_dir)
    write_parquet_or_pickle(predictions, Path(args.output_dir) / "window_predictions.parquet")
    metrics.to_csv(Path(args.output_dir) / "window_metrics.csv", index=False)
    print(metrics.groupby("model_muscle")[["mae", "rmse", "r2"]].mean().to_string())
    return 0


def _extract_features(args: argparse.Namespace) -> int:
    output = extract_feature_dataset(
        args.data_root,
        args.output,
        overwrite=args.overwrite,
    )
    print(output)
    return 0


def _rebuild_models(args: argparse.Namespace) -> int:
    table = read_table(args.features)
    reference = Path(args.reference_model) if args.reference_model else None
    paths = rebuild_published_models(
        table,
        args.output_dir,
        reference_model=reference,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    if args.predictions:
        source = Path(args.predictions)
    elif args.model:
        source = Path(args.model)
        if source.is_dir():
            source = source / "window_predictions.parquet"
    else:
        raise ValueError("provide a predictions table or --model output directory")
    frame = read_table(source)
    required = {args.reference_column, args.predicted_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    metrics = window_metrics(
        frame[args.reference_column].to_numpy(dtype=float),
        frame[args.predicted_column].to_numpy(dtype=float),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the complete command-line parser."""

    parser = argparse.ArgumentParser(
        prog="emg-normalisation",
        description="Distribution-informed EMG normalisation and model validation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data", help="Validate bundled public data and manifests.")
    validate.add_argument("--data-root", default="data")
    validate.add_argument("--example-root", default="examples/full_trial_data/C06/trial_ffc10db23f")
    validate.set_defaults(function=_validate_data)

    reference = sub.add_parser(
        "build-reference",
        help="Build conventional reference-normalised envelopes from bandpass EMG.",
    )
    reference.add_argument("--data-root", default="data")
    reference.set_defaults(function=_build_reference)

    predict = sub.add_parser("predict", help="Apply the published model to one bandpass EMG file.")
    predict.add_argument("input", help="OpenSim-style emgBPF.mot file.")
    predict.add_argument("--model", default=None, help="Published model.json; defaults to bundled model.")
    predict.add_argument("--output", required=True, help="Output Parquet path.")
    predict.add_argument(
        "--gain",
        type=float,
        help="Known amplifier voltage gain; required for any stored_* input unit.",
    )
    predict.add_argument(
        "--input-unit",
        choices=("stored_v", "stored_mV", "stored_uV", "mV"),
        default="stored_v",
        help=(
            "Unit of amplified stored values, or mV for input-referred data "
            "that already have gain removed."
        ),
    )
    predict.add_argument(
        "--output-mode",
        choices=("scaled_envelope", "model_trace"),
        default="scaled_envelope",
    )
    predict.set_defaults(function=_predict)

    train = sub.add_parser("train", help="Run fixed LOSO training with the manuscript or a custom model.")
    train.add_argument("--features", default="data/derived/window_features.parquet")
    train.add_argument(
        "--model",
        default="linear",
        help=(
            "Built-in model ("
            + ", ".join(BUILT_IN_MODELS)
            + ") or a custom factory using path/to/module.py:function."
        ),
    )
    train.add_argument("--output-dir", default="outputs/model_loso")
    train.set_defaults(function=_train)

    extract = sub.add_parser(
        "extract-features",
        help="Build window variance, KDE FWHM, and reference-target rows from signals.",
    )
    extract.add_argument("--data-root", default="data")
    extract.add_argument("--output", default="data/derived/window_features.parquet")
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(function=_extract_features)

    rebuild = sub.add_parser(
        "rebuild-published-models",
        help="Run C01-C07 LOSO validation and rebuild the final linear models.",
    )
    rebuild.add_argument("--features", default="data/derived/window_features.parquet")
    rebuild.add_argument("--output-dir", default="outputs/published_model_rebuild")
    rebuild.add_argument(
        "--reference-model",
        default="models/cycling_linear/model.json",
        help="Released model.json to compare against; use an empty value to skip comparison.",
    )
    rebuild.set_defaults(function=_rebuild_models)

    evaluate = sub.add_parser(
        "evaluate",
        help="Calculate shared window-level metrics for a prediction table.",
    )
    evaluate.add_argument(
        "predictions",
        nargs="?",
        help="CSV or Parquet containing targets and predictions.",
    )
    evaluate.add_argument(
        "--model",
        help="Training output directory containing window_predictions.parquet.",
    )
    evaluate.add_argument("--reference-column", default="reference_normalised_target")
    evaluate.add_argument("--predicted-column", default="prediction")
    evaluate.add_argument("--output", default="outputs/evaluation/window_metrics.json")
    evaluate.set_defaults(function=_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the selected command and return a process status."""

    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
