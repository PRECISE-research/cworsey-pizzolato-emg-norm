# Getting started

## Install

```bash
git clone https://github.com/PRECISE-research/carrall-worsey-pizzolato-emg-normalisation.git
cd carrall-worsey-pizzolato-emg-normalisation
git lfs pull --include="examples/full_trial_data/**,data/derived/window_features.parquet"
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## First prediction

```python
from emg_normalisation import load_published_model, predict_trial, read_emg_mot
from emg_normalisation.calibration import removeGain

stored = read_emg_mot("examples/full_trial_data/C06/trial_ffc10db23f/emgBPF.mot")
input_mV = removeGain(stored, amplifier_gain=1000)
bundle = load_published_model("models/cycling_linear/model.json")
result = predict_trial(input_mV, bundle, output_mode="scaled_envelope")
result.output.to_csv("outputs/example_prediction.csv", index=False)
```

The output is dimensionless normalised excitation. Input signals passed to
`predict_trial` must already have amplifier gain removed and be expressed in
input-referred mV.

The bundled real-data recording is a complete anonymised C06 cycling trial. It
includes both sides of all 13 anatomical muscles used by the published model,
so the first prediction uses the same format as the manuscript workflow.

## Which workflow should I use?

| Goal | Command/API |
|---|---|
| Apply the paper model | `emg-normalisation predict` |
| Return the paper scaled envelope | `output_mode="scaled_envelope"` |
| Return direct interpolated predictions | `output_mode="model_trace"` |
| Rebuild reference envelopes from full data archive | `emg-normalisation build-reference` |
| Train another estimator from bundled real feature cache | `emg-normalisation train --model file.py:factory` |
| Repeat C01-C07 LOSO and rebuild the linear models | `emg-normalisation rebuild-published-models` |
