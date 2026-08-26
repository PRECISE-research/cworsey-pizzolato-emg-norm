# Model provenance

- Development cohort: `C01`-`C07`.
- Training trials: 17 primary cycling conditions and three
  countermovement-jump trials per participant.
- External evaluation: `E01`; never used to fit the published models.
- Model scope: one side-pooled model per anatomical muscle.
- Inputs: bandpass-filtered EMG with known amplifier gain removed, expressed in
  input-referred mV.
- Features: population variance (mV²) and Gaussian-KDE FWHM (mV).
- Windowing: 0.100-s windows, 0.025-s step, at least 0.030 s of valid data.
- Targets: conventional reference-normalised envelope at the window centre.
- Conventional envelope: full-wave rectification and second-order, zero-phase
  6-Hz Butterworth low-pass filtering.
- Residual floor: none removed from the model trace, conventional envelope, or
  reference envelope.
- Final reconstruction: the maximum bounded and interpolated model prediction
  scales the conventional envelope once per trial.

Reference selections and excluded channels are machine-readable in
`data/manifests/`.
