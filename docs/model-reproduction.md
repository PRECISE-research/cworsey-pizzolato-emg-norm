# Model reproduction

The repository ships the published 13-muscle cycling model, the deidentified
C01-C07 feature/target cache, and the code needed to repeat validation and
fitting. The raw cohort EMG is not included.

## Required input

The included feature table is at `data/derived/window_features.parquet`. Its
schema is documented in
[Data format](data-format.md). The training command enforces all of the
following before fitting:

- participant is C01-C07;
- `include_training` is true;
- `qc_status` is `included`;
- left and right channels are pooled by `model_muscle`;
- features are `variance_mV2` and `kde_fwhm_mV`;
- target is `reference_normalised_target`.

The bundled feature table is the complete public input required for validation
and model fitting. It intentionally uses opaque participant and trial labels;
no mapping to source-study identifiers is included.

## Repeat LOSO and rebuild the final models

```bash
emg-normalisation rebuild-published-models \
  --features data/derived/window_features.parquet \
  --output-dir outputs/published_model_rebuild
```

The command first fits 13 muscle-specific models in each leave-one-participant-
out fold. It then fits the published model form to all eligible rows from C01-C07.
Nothing under `models/cycling_linear/` is overwritten.

Outputs are:

| Path | Contents |
|---|---|
| `loso_models/` | One fitted model artifact per held-out participant and muscle |
| `loso_window_predictions.parquet` | Held-out window targets and predictions |
| `loso_window_metrics.csv` | Fold-by-muscle MAE, RMSE, and R2 |
| `model.json` | Rebuilt final 13-muscle model bundle |
| `coefficients.csv` | Human-readable standardisation values and coefficients |
| `model.pkl` | Python serialisation of the rebuilt bundle |
| `published_model_comparison.csv` | Exact parameter differences from the bundled JSON |
| `rebuild_summary.json` | Participants, row counts, model count, and output paths |

Inspect `published_model_comparison.csv` before replacing or versioning any
bundled artifact. Small floating-point differences can occur across numerical
library versions; changes in included rows, muscle count, or material parameter
values require investigation.

## Validate another estimator

Use the same feature table and folds with:

```bash
emg-normalisation train \
  --features data/derived/window_features.parquet \
  --model hist-gradient-boosting \
  --output-dir outputs/hist-gradient-boosting
```

Choose `linear`, `ridge`, `hist-gradient-boosting`, or `random-forest`, or pass
`examples/custom_model.py:create_model` to use the custom-estimator interface.
All learned preprocessing and model selection must be fitted without using the
held-out participant.
