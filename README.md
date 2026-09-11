# Carrall-Worsey-Pizzolato EMG normalisation

[![CI](https://github.com/PRECISE-research/carrall-worsey-pizzolato-emg-normalisation/actions/workflows/ci.yml/badge.svg)](https://github.com/PRECISE-research/carrall-worsey-pizzolato-emg-normalisation/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/Code%20license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-teal.svg)](pyproject.toml)

Reproducible implementation of a distribution-informed method for assigning a
normalised amplitude scale to conventional surface-EMG linear envelopes without
requiring a maximal calibration task at the point of application.

The method extracts **window variance** and Gaussian kernel-density-estimate
**full width at half maximum (KDE FWHM)** from gain-corrected bandpass EMG. A
side-pooled, anatomical-muscle linear model estimates normalised excitation at
each window centre. The manuscript implementation uses the maximum interpolated
estimate to scale the conventional 6-Hz linear envelope, preserving its measured
temporal profile.

## What is included

- One anonymised real C06 cycling trial, tracked with Git LFS, for transparent
  worked examples.
- A deidentified C01-C07 window-feature and reference-target cache, tracked with
  Git LFS, for training and comparing alternative models.
- The final 13 cycling-cohort regression models in JSON and CSV.
- A command that repeats fixed C01-C07 leave-one-participant-out (LOSO)
  validation and rebuilds the final all-participant linear models.
- Fixed LOSO and QC manifests.
- An API for using the published model or fitting any estimator with
  `fit(features, targets)` and `predict(features)`.
- Automated tests using synthetic fixtures; they do not require raw recordings.

The complete raw EMG archive, all-trial reference envelopes, segmentation files,
figure-ready tables, rendered figures, editable manuscript Word documents, and
participant-code mapping are not included. The included feature cache contains
no identifying information or raw EMG samples.

## Installation

```bash
git clone https://github.com/PRECISE-research/carrall-worsey-pizzolato-emg-normalisation.git
cd carrall-worsey-pizzolato-emg-normalisation
git lfs pull --include="examples/full_trial_data/**,data/derived/window_features.parquet"
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Five-minute quick start

```python
from emg_normalisation import load_published_model, predict_trial, read_emg_mot
from emg_normalisation.calibration import removeGain

bandpass = read_emg_mot(
    "examples/full_trial_data/C06/trial_ffc10db23f/emgBPF.mot"
)
# Supply the known voltage gain of the recording amplifier.
bandpass_mV = removeGain(bandpass, amplifier_gain=1000)
model = load_published_model("models/cycling_linear/model.json")

# Manuscript method: preserve the conventional-envelope shape.
scaled = predict_trial(
    bandpass_mV,
    model,
    output_mode="scaled_envelope",
).output

# Alternative: return the interpolated window-centre model estimates directly.
direct = predict_trial(
    bandpass_mV,
    model,
    output_mode="model_trace",
).output
```

The bundled real-data example is a complete anonymised C06 maximal-effort
isokinetic cycling trial at 110 rpm with all 26 bilateral channels representing
the 13 anatomical-muscle models. All public signal inputs are in
**input-referred mV** after amplifier-gain removal. Variance is therefore in
mV2 and KDE FWHM in mV.

## Apply the model to your own EMG

The dedicated user-data example accepts OpenSim MOT, CSV, CSV.GZ, or Parquet
bandpass EMG:

```bash
python examples/predict_own_emg.py my_bandpass.mot outputs/my_estimate.csv \
  --amplifier-gain 1000
```

Use `--already-input-mv` instead when amplifier gain has already been removed.
Use `--channel-map examples/channel_map_template.json` when your channel names
differ from the documented names. See
[Using your own data](docs/using-your-own-data.md) for the full input contract.

## Published model

```python
model = load_published_model("models/cycling_linear/model.json")
vaslat = model.models["vaslat"]
print(vaslat.intercept)
print(vaslat.coefficients)
print(vaslat.feature_mean)
print(vaslat.feature_sd)
```

Left and right channels map to the same anatomical-muscle model while predictions
remain side specific.

## Train alternative models

The deidentified C01-C07 feature cache at
`data/derived/window_features.parquet` contains real study features and
reference-normalised targets. It is the shared input for all public model
comparisons.

Try a built-in alternative without writing code:

```bash
emg-normalisation train \
  --features data/derived/window_features.parquet \
  --model hist-gradient-boosting \
  --output-dir outputs/hist-gradient-boosting
```

Built-in choices are `linear`, `ridge`, `hist-gradient-boosting`, and
`random-forest`. Run each into a separate output directory to compare their
held-out predictions and metrics under identical C01-C07 LOSO splits.

Create a factory returning any estimator with `fit` and `predict`:

```python
# my_models.py
from sklearn.ensemble import HistGradientBoostingRegressor
from emg_normalisation.models import SklearnEstimatorAdapter

def create_model():
    return SklearnEstimatorAdapter(
        HistGradientBoostingRegressor(random_state=1)
    )
```

Run the same fixed LOSO splits used by the common evaluator:

```bash
emg-normalisation train \
  --features data/derived/window_features.parquet \
  --model my_models.py:create_model \
  --output-dir outputs/hist_gradient_boosting
```

The repository does not prescribe or pre-train alternative model types. The
included feature cache, folds, reconstruction, and metrics keep comparisons
scientifically meaningful.

## Rebuild the published linear models

The public code includes the deidentified feature cache required to repeat the
model-fitting route. Run:

```bash
emg-normalisation rebuild-published-models \
  --features data/derived/window_features.parquet \
  --output-dir outputs/published_model_rebuild
```

This applies the audited training and channel exclusions, holds out each of
C01-C07 in turn, exports LOSO predictions and metrics, then fits one final
side-pooled linear model per anatomical muscle using all seven participants.
The rebuilt JSON and CSV are compared parameter-by-parameter with the bundled
model. See [Model reproduction](docs/model-reproduction.md).

## Repository guide

| Path | Purpose |
|---|---|
| `src/emg_normalisation` | Installable signal-processing and modelling API |
| `data/derived` | Deidentified C01-C07 feature and target cache for model development |
| `data/manifests` | Public metadata describing trial roles, units, checksums, reference maxima, and QC |
| `models/cycling_linear` | Published cycling-cohort model bundle |
| `examples` and `notebooks` | Executable tutorials |
| `docs` | Detailed methods and API documentation |

## Documentation

Install and serve the complete documentation site:

```bash
pip install -e .[docs]
mkdocs serve
```

Start with [Getting started](docs/getting-started.md), then see the
[API reference](docs/api-reference.md) and
[custom-model guide](docs/training-custom-models.md).

## Testing

```bash
pip install -e .[dev]
pytest
mkdocs build --strict
```

CI downloads only the bundled real-data example trial; model-development data
remain an explicit Git LFS download.

## Data governance

The code is Apache-2.0 licensed. The bundled real-data example, deidentified
feature cache are subject to the research-use conditions in
[DATA_USE.md](DATA_USE.md). Confirm institutional requirements before using
human-derived data.

## Citation

Please cite the associated preprint and this repository:

> Carrall-Worsey, M., & Pizzolato, C. (2026). *Normalising electromyograms
> without maximal voluntary contractions*. Research Square.
> https://doi.org/10.21203/rs.3.rs-10721340/v1

Machine-readable metadata are provided in [CITATION.cff](CITATION.cff).
