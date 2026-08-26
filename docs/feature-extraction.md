# Feature extraction

The input signal is divided into 100-ms windows advanced every 25 ms. A window
must contain at least 30 ms of finite samples.

```python
from emg_normalisation.features import extract_window_features

features = extract_window_features(
    signal_mV,
    sampling_rate_hz=2000,
    window_s=0.100,
    step_s=0.025,
    min_valid_s=0.030,
)
```

## Variance

Population variance describes the overall dispersion of the local bandpass EMG
amplitudes. With mV input, its unit is mV2.

## KDE FWHM

Gaussian kernel density estimation provides a smooth non-parametric estimate of
the amplitude distribution. SciPy's Scott bandwidth rule is used. The kernel
being Gaussian does not assert that EMG amplitudes are Gaussian.

The density is evaluated on 120 equally spaced locations from the window
minimum to maximum. FWHM is the distance between the outermost locations at or
above half the density peak. With mV input, FWHM is in mV.

Variance and KDE FWHM are standardised using training-fold means and SDs before
model fitting. Held-out data never contribute to those statistics.
