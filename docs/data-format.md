# Data format

## Bandpass EMG

This repository includes one real example under `examples/full_trial_data/`.
Other raw recordings use the same formats described below.

Files use an OpenSim-style whitespace-delimited `.mot` layout:

```text
Bandpass EMG
nRows=...
nColumns=...

endheader
time    r_vaslat    r_vasmed    ...
```

- `time` is seconds.
- Signal columns are side-specific muscle channels.
- `.mot` values are acquisition-system volts.
- The modelling API expects amplifier gain removed and values converted to mV.
- Cycling files contain bilateral channels where available.
- The external participant contains right-side channels.

## Participant labels

`C01` to `C07` are the cycling development cohort. `E01` is the
independent non-cycling participant.

## Reference envelopes

In the real example and during feature generation,
`reference_normalised_envelope.parquet` contains:

- time in seconds;
- one dimensionless 0-1-reference channel per valid EMG channel;
- no residual-floor subtraction.

Values may exceed one outside the trial containing the participant-muscle
reference maximum only if filtering/interpolation creates a numerical overshoot;
the stored reference targets are otherwise left unmodified.

## Window feature table

The bundled deidentified Parquet feature table contains:

| Column | Meaning |
|---|---|
| `participant` | C01-C07 |
| `trial` | Trial identifier |
| `channel` | Side-specific channel |
| `model_muscle` | Side-pooled anatomical muscle |
| `centre_index` | Window-centre sample |
| `centre_time_s` | Window-centre time in seconds |
| `variance_mV2` | Population variance in mV2 |
| `kde_fwhm_mV` | KDE FWHM in mV |
| `reference_normalised_target` | Reference envelope at the window centre |
| `loso_fold` | Participant held out in the corresponding fold |
| `include_training` | Eligible for fixed cycling-cohort model development |
| `include_primary_reporting` | Primary cycling reporting trial flag |
| `qc_status` | Final channel/window inclusion status |

The bundled cache contains the seven cycling participants only. `E01`
and the raw signal files are not included.

## Manifests

`data/manifests` records trial roles, checksums, reference maxima, calibration,
cadence mapping, and exclusions. These concise files are retained as
analysis-definition metadata.

### Which trial file to read

Two files describe the same 246 trials, and they are not interchangeable.

| File | Scope | Read it for |
|---|---|---|
| `trial_roles.csv` | Full role table, 12 columns | Dataset assembly. This is the file the package reads (`emg_normalisation.dataset`) |
| `trial_manifest.csv` | Condensed summary, 7 columns | A quick inventory of participants, trials, and top-level inclusion flags |

`trial_roles.csv` is authoritative. Where the two overlap, these column names
correspond:

| `trial_manifest.csv` | `trial_roles.csv` |
|---|---|
| `include_reference_normalisation` | `include_reference` |
| `include_model_training` | `include_training` |
| `external_primary_evaluation` | `nl_external_evaluation` |

Those three agree row-for-row across all 246 trials.

The two files also use different vocabularies for the `E01` trial roles:
`trial_manifest.csv` uses `external_dynamic_reference` and
`external_mvic_reference` where `trial_roles.csv` uses
`nl_dynamic_augmentation` and `nl_mvic_augmentation`. These name the same 106
rows. No `C01`-`C07` trial role differs between the files.
