# Troubleshooting

## Predictions are unexpectedly high or low

Confirm the input is in input-referred mV after amplifier-gain removal. Passing
stored acquisition volts, microvolts, or unknown-gain values changes variance
and KDE FWHM and invalidates the published feature statistics.

## Parquet support is unavailable

Install `pyarrow`:

```bash
pip install pyarrow
```

## Git LFS files are tiny text pointers

```bash
git lfs install
git lfs pull --include="examples/full_trial_data/**,data/derived/window_features.parquet"
emg-normalisation validate-data
```

The C01-C07 feature cache and C06 example are Git LFS files. Raw recordings are
not part of this repository.

## A muscle is missing

The published bundle supports 13 anatomical muscles and expects the channel
names documented in `data/manifests/channel_dictionary.csv`. The final QC
manifest also excludes five participant-channel combinations.

## Sampling rate differs from 2000 Hz

The API expresses window settings in seconds and converts them to samples using
the supplied time vector. Preserve 100-ms windows, 25-ms updates, and a 30-ms
minimum valid duration.

## A custom model leaks held-out data

Use `emg_normalisation.training.loso_splits`. Any feature standardisation,
imputation, or hyperparameter selection must be fitted on the training
participants only.

## Rebuilding the feature cache is slow

The included cache is ready for model fitting. Rebuilding it from signal-level
files intentionally repeats KDE extraction across millions of windows.
