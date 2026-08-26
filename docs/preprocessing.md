# Preprocessing

## Gain removal

The model was developed from input-referred electrode voltage. For any
amplifier system:

```text
input mV = stored V x 1000 / amplifier gain
```

The amplifier gain must be known and supplied explicitly. The manuscript
recordings used a gain of 1000. The archived Vicon SourceGain is treated as
already applied by Nexus before `.mot` export. Applying the published
coefficients to incompatible units will invalidate variance and KDE FWHM.

```python
from emg_normalisation.calibration import removeGain
input_mV = removeGain(stored_frame, amplifier_gain=1000)
```

`removeGain` is hardware agnostic and also accepts stored mV or µV through its
`stored_unit` argument. It always returns input-referred mV.

## Conventional envelope

The reference and final scaled waveform use:

1. gain-corrected 30-300 Hz bandpass EMG;
2. full-wave rectification;
3. second-order, zero-phase 6-Hz Butterworth low-pass filtering;
4. no residual-floor subtraction.

```python
from emg_normalisation.preprocessing import conventional_linear_envelope
envelope_mV = conventional_linear_envelope(input_mV)
```

## Reference normalisation

Each participant-muscle envelope is divided by its audited maximum across the
designated reference trials. Cycling participants use the 17 cycling trials and
three CMJs. The external participant uses all supplied dynamic and MVIC trials,
with the documented tibialis-anterior MVIC override.

The maxima table records the source trial, time, value, and selection mode.
