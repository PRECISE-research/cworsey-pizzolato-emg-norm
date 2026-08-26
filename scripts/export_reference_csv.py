"""Export one or all compressed reference envelopes to ordinary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Reference-envelope Parquet file or directory.")
    parser.add_argument("--output-dir", default="outputs/reference_csv")
    args = parser.parse_args()
    source = Path(args.input)
    files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    output_root = Path(args.output_dir)
    for path in files:
        relative = path.name if source.is_file() else path.relative_to(source).with_suffix(".csv")
        output = output_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(path).to_csv(output, index=False)
        print(output)


if __name__ == "__main__":
    main()
