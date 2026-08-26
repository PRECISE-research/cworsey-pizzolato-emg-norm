"""Apply the published model to a complete real-data example trial."""

from pathlib import Path

from emg_normalisation import load_published_model, predict_trial, read_emg_mot
from emg_normalisation.calibration import removeGain

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "full_trial_data" / "C06" / "trial_ffc10db23f" / "emgBPF.mot"
MODEL = ROOT / "models" / "cycling_linear" / "model.json"
OUTPUT = ROOT / "outputs" / "quick_start_prediction.csv"


def main() -> None:
    stored_volts = read_emg_mot(INPUT)
    input_mV = removeGain(stored_volts, amplifier_gain=1000)
    result = predict_trial(input_mV, load_published_model(MODEL), output_mode="scaled_envelope")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.output.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
