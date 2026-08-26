"""Input/output helpers for EMG and repository datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


def read_emg_mot(path: str | Path) -> pd.DataFrame:
    """Read a whitespace-delimited OpenSim-style bandpass EMG file.

    Parameters
    ----------
    path:
        File containing an ``endheader`` line followed by a column-name row.
        The first column must be ``time`` in seconds. EMG channels are stored
        in acquisition-system volts and require gain removal before modelling.

    Returns
    -------
    pandas.DataFrame
        Time and EMG channels in the order stored in the file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the header or time column is missing.

    Examples
    --------
    >>> frame = read_emg_mot("examples/full_trial_data/C06/trial_ffc10db23f/emgBPF.mot")
    >>> frame.columns[0]
    'time'
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    header_end: int | None = None
    columns: list[str] | None = None
    with file_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if line.strip().lower() == "endheader":
                header_end = line_number
                columns = next(stream, "").split()
                break
    if header_end is None or not columns:
        raise ValueError(f"No OpenSim endheader/column row found in {file_path}")
    frame = pd.read_csv(
        file_path,
        sep=r"\s+",
        engine="c",
        skiprows=header_end + 2,
        names=columns,
        header=None,
    )
    if "time" not in frame.columns:
        raise ValueError(f"No time column found in {file_path}")
    return frame


def write_emg_mot(frame: pd.DataFrame, path: str | Path, name: str = "Bandpass EMG") -> Path:
    """Write an OpenSim-style EMG ``.mot`` file.

    Parameters
    ----------
    frame:
        DataFrame whose first column is time in seconds.
    path:
        Destination path.
    name:
        Human-readable file name included in the header.

    Returns
    -------
    pathlib.Path
        The written path.
    """

    if "time" not in frame.columns:
        raise ValueError("frame must include a 'time' column")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{name}\n")
        stream.write(f"nRows={len(frame)}\n")
        stream.write(f"nColumns={len(frame.columns)}\n\n")
        stream.write("endheader\n")
        stream.write("\t".join(frame.columns) + "\n")
        frame.to_csv(stream, sep="\t", index=False, header=False, float_format="%.10g")
    return output


def iter_trial_files(data_root: str | Path) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(participant, trial, emg_path)`` records in stable order."""

    root = Path(data_root)
    for participant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for trial_dir in sorted(path for path in participant_dir.iterdir() if path.is_dir()):
            emg_path = trial_dir / "emgBPF.mot"
            if emg_path.exists():
                yield participant_dir.name, trial_dir.name, emg_path


def write_parquet_or_pickle(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write compressed Parquet, with a documented pickle fallback.

    Parquet is preferred for interoperable, compressed arrays. The fallback is
    used only when no Parquet engine is installed and appends ``.pkl``.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(output, index=False, compression="zstd")
        return output
    except ImportError:
        fallback = output.with_suffix(output.suffix + ".pkl")
        frame.to_pickle(fallback)
        return fallback


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV, CSV.GZ, Parquet, or pickle data by extension."""

    file_path = Path(path)
    lower = file_path.name.lower()
    if lower.endswith((".csv", ".csv.gz")):
        return pd.read_csv(file_path)
    if lower.endswith(".parquet"):
        return pd.read_parquet(file_path)
    if lower.endswith((".pkl", ".pickle")):
        return pd.read_pickle(file_path)
    raise ValueError(f"Unsupported table format: {file_path}")
