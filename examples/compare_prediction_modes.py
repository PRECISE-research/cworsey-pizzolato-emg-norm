"""Compare model outputs with the supplied reference envelope for one channel."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from emg_normalisation import load_published_model, predict_trial, read_emg_mot
from emg_normalisation.calibration import removeGain
from emg_normalisation.io import read_table

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    trial_root = ROOT / "examples" / "full_trial_data" / "C06" / "trial_ffc10db23f"
    frame = read_emg_mot(trial_root / "emgBPF.mot")
    frame_mV = removeGain(frame, amplifier_gain=1000)
    reference = read_table(trial_root / "reference_normalised_envelope.parquet")
    model = load_published_model(ROOT / "models/cycling_linear/model.json")
    result = predict_trial(frame_mV, model, output_mode="scaled_envelope")

    channel = "r_vaslat"
    figure, axis = plt.subplots(figsize=(7, 3.2))
    axis.plot(result.output["time"], result.output[channel], alpha=0.8, label="Trial-scaled envelope")
    axis.plot(
        result.model_trace["time"],
        result.model_trace[channel],
        linestyle="--",
        alpha=0.8,
        label="Direct model trace",
    )
    axis.plot(
        reference["time"],
        reference[channel],
        color="black",
        linewidth=1.2,
        zorder=3,
        label="Reference envelope",
    )
    axis.set(xlabel="Time (s)", ylabel="Normalised excitation", ylim=(0, 1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "prediction_modes_vs_reference.png"
    figure.savefig(figure_path, dpi=300)

    comparison = pd.DataFrame(
        {
            "time": result.output["time"],
            "reference_envelope": reference[channel],
            "scaled_envelope": result.output[channel],
            "direct_model_trace": result.model_trace[channel],
        }
    )
    table_path = output_dir / "prediction_modes_vs_reference.csv"
    comparison.to_csv(table_path, index=False)
    print(f"Saved {figure_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
