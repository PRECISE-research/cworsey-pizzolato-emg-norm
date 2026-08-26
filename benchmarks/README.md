# Alternative-model benchmarks

The bundled cohort feature table and fixed folds allow alternative regression
methods to be evaluated without reimplementing EMG processing.

Start with a built-in nonlinear model:

```bash
emg-normalisation train \
  --features data/derived/window_features.parquet \
  --model hist-gradient-boosting \
  --output-dir outputs/hist-gradient-boosting
```

Available built-ins are `linear`, `ridge`, `hist-gradient-boosting`, and
`random-forest`. Use the custom template below when another estimator is
needed.

```bash
emg-normalisation train \
  --features data/derived/window_features.parquet \
  --model benchmarks/custom_model_template.py:create_model \
  --output-dir outputs/example_model
```

The template intentionally uses a simple scikit-learn estimator. Replace it
with any object implementing `fit` and `predict`. Hyperparameter selection must
be confined to training participants, ideally with nested cross-validation.

For fair manuscript comparisons:

1. Retain the supplied QC exclusions and C01-C07 folds.
2. Train one model per side-pooled anatomical muscle.
3. Bound reconstructed amplitude estimates to 0-1.
4. Report cycle-level absolute peak error as the primary endpoint.
5. Report both direct `model_trace` and manuscript `scaled_envelope` outputs
   when waveform reconstruction is relevant.
