# Evaluation

## Cycle-level amplitude error

For cycle `c`:

```text
signed peak error (%) = 100 x (reference peak - predicted peak)
absolute peak error (%) = abs(signed peak error)
```

Positive signed error indicates underestimation; negative error indicates
overestimation. Errors are calculated for each crank or movement cycle and
averaged within trial before participant, condition, and muscle aggregation.

## Cycling

Cycling is segmented to 361 points representing 0-360 degrees. Right channels
use 180-degree crossings. Left channels use wrapped-angle resets so both limbs
have top dead centre at 0 degrees.

Randomised maximal-isokinetic trial order is resolved through the participant
cadence manifest before condition aggregation.

## External tasks

Walking, running, squat, and CMJ repetitions are time-normalised to 101 points.
The external participant is evaluated using the final model trained on all
cycling participants; external data do not enter model fitting.

## Window diagnostics

The common evaluator also reports window-centre MAE, RMSE, and R2. These measure
the direct regression trace and are not equivalent to cycle-level peak error.
# Evaluating trained outputs

`train` writes `window_predictions.parquet` and `window_metrics.csv`. Recompute
the shared window metrics directly from the output directory:

```bash
emg-normalisation evaluate --model outputs/my_model
```

or from any compatible prediction table:

```bash
emg-normalisation evaluate predictions.parquet
```
