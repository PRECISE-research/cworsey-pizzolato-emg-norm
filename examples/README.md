# Examples

The `full_trial_data/` example is a complete anonymised real C06 maximal-effort
isokinetic cycling trial at 110 rpm. It contains all 26 bilateral EMG channels
for the 13 anatomical muscles and the corresponding conventional
reference-normalised envelopes in Parquet format. The signal and Parquet files
are tracked through Git LFS.

```bash
python examples/quick_start.py
python examples/compare_prediction_modes.py
```

- `predict_own_emg.py` is the simplest entry point for a user's own bandpass
  EMG. It accepts MOT, CSV, CSV.GZ, or Parquet input and records gain handling
  and channel mapping in a metadata file.
- `quick_start.py` applies the published cycling model in manuscript
  `scaled_envelope` mode.
- `compare_prediction_modes.py` plots the supplied reference envelope beside
  the trial-scaled waveform and direct interpolated `model_trace`, and saves
  the three traces to CSV.
- `custom_model.py` demonstrates the estimator-factory interface.

Users who do not need a custom factory can select `linear`, `ridge`,
`hist-gradient-boosting`, or `random-forest` directly with
`emg-normalisation train --model NAME`.

The complete tutorials in `notebooks/` expand these operations step by step.

## Use your own bandpass EMG

For amplified EMG stored in volts:

```bash
python examples/predict_own_emg.py my_bandpass.mot outputs/my_estimate.csv \
  --amplifier-gain 1000
```

For EMG already expressed as input-referred mV:

```bash
python examples/predict_own_emg.py my_bandpass.csv outputs/my_estimate.csv \
  --already-input-mv
```

The input must contain time in seconds and canonical side-specific muscle names
such as `r_vaslat` or `l_vaslat`. If names differ, copy
`channel_map_template.json`, edit the mappings, and add:

```bash
--channel-map my_channel_map.json
```
