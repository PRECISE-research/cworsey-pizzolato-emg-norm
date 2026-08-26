# Published models

The published bundle contains one multiple linear least-squares model for each
of 13 anatomical muscles. Left and right channels are pooled during fitting:

```text
y_hat = beta0 + beta_variance z_variance + beta_FWHM z_FWHM
```

where each `z` uses the muscle-specific cycling-training mean and SD.

```python
from emg_normalisation import load_published_model

bundle = load_published_model("models/cycling_linear/model.json")
model = bundle.models["vaslat"]
prediction = model.predict([[variance_mV2, kde_fwhm_mV]])
```

## Intended use

- Compatible gain-removed surface EMG.
- Matching bandpass and window definitions.
- Estimating the study-derived excitation scale represented by low-power
  cycling, maximal isokinetic cycling, and CMJ data.

## Limitations

The coefficients are not hardware agnostic. Electrode type, inter-electrode
distance, placement, skin preparation, acquisition hardware, filtering, and
unknown gain can alter distribution features. The model should not be treated
as an independent physiological gold standard.
