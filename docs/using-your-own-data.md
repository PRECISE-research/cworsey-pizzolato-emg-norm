# Using your own bandpass EMG

The simplest entry point is `examples/predict_own_emg.py`. It applies the
published cycling-cohort model to one complete trial without requiring users to
write package code.

## Required input

Supply an OpenSim-style MOT, CSV, CSV.GZ, or Parquet file containing:

- one `time` column in seconds;
- gain-removed bandpass EMG in input-referred mV, or amplified voltage with its
  known amplifier gain;
- at least one recognised side-specific muscle channel.

The script expects EMG that has already undergone the study-compatible
30-300 Hz bandpass filtering. It does not bandpass-filter raw EMG.

## Amplified stored voltage

If values remain amplified, provide the known dimensionless voltage gain:

```bash
python examples/predict_own_emg.py data/my_bandpass.mot outputs/my_estimate.csv \
  --amplifier-gain 1000 \
  --stored-unit V
```

`--stored-unit` may be `V`, `mV`, or `uV`. The script removes the supplied gain
and converts the input to input-referred mV.

## Gain already removed

```bash
python examples/predict_own_emg.py data/my_bandpass.csv outputs/my_estimate.csv \
  --already-input-mv
```

Only use this flag when values are already input-referred mV.

## Channel names

Published channel names use `r_` and `l_` prefixes:

| Muscle | Channel stem |
|---|---|
| Vastus lateralis | `vaslat` |
| Vastus medialis | `vasmed` |
| Rectus femoris | `recfem` |
| Adductor | `add` |
| Tibialis anterior | `tibant` |
| Semitendinosus | `semiten` |
| Biceps femoris long head | `bflh` |
| Medial gastrocnemius | `medgas` |
| Lateral gastrocnemius | `latgas` |
| Soleus | `soleus` |
| Gluteus maximus | `glmax` |
| Gluteus medius | `glmed` |
| Tensor fasciae latae | `tfl` |

For example, right vastus lateralis is `r_vaslat`. Left and right channels use
the same side-pooled anatomical-muscle model but produce separate envelopes.

If input names differ, create a JSON mapping:

```json
{
  "Right VL": "r_vaslat",
  "Left VL": "l_vaslat"
}
```

Then run with `--channel-map my_channel_map.json`.

## Output modes

The default `scaled_envelope` mode reproduces the manuscript implementation:

```bash
--output-mode scaled_envelope
```

It scales the conventional rectified, 6-Hz low-pass envelope to the maximum
normalised amplitude estimated by the model for the complete trial.

The alternative mode returns interpolated model estimates directly:

```bash
--output-mode model_trace
```

## Outputs

The requested CSV contains time and one normalised 0-1 excitation estimate per
recognised channel. A neighbouring `.metadata.json` records:

- model and input paths;
- amplifier-gain handling;
- channel mapping;
- predicted and skipped channels;
- estimated trial peak for every predicted channel;
- selected output mode.

Run `python examples/predict_own_emg.py --help` for all options.
