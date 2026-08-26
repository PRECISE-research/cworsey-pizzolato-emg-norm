# Prediction modes

The regression model estimates normalised amplitude at each 25-ms window centre.
Predictions are constrained to 0-1 and reconstructed with cubic spline
interpolation when at least four unique centres are available; otherwise linear
interpolation is used.

## Scaled envelope

```python
result = predict_trial(frame_mV, bundle, output_mode="scaled_envelope")
```

This reproduces the manuscript:

1. reconstruct the direct model trace;
2. take its complete-trial maximum;
3. build the conventional rectified 6-Hz envelope;
4. multiply the conventional envelope by
   `estimated_peak / conventional_peak`.

The temporal shape is therefore the measured conventional envelope.

## Model trace

```python
result = predict_trial(frame_mV, bundle, output_mode="model_trace")
```

This returns the interpolated regression predictions themselves. It can be
useful when testing whether a model predicts temporal shape as well as
amplitude, but it is not the scaled-envelope implementation used for the
manuscript headline results.

`PredictionResult` always contains both representations, allowing direct
comparison without extracting features twice.
