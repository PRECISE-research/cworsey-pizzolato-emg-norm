# Deidentified C01-C07 model-development cache

`window_features.parquet` is the real, deidentified feature-and-target dataset
used to train and validate the published cycling models.

- Rows: 4,512,263
- Participants: opaque labels C01-C07
- Trials: 20 per participant (17 cycling conditions and three CMJ augmentation
  trials)
- Muscles: 13 side-pooled anatomical muscles
- Inputs: window variance (`variance_mV2`) and KDE FWHM (`kde_fwhm_mV`)
- Target: clipped 0-1 conventional reference-normalised envelope value at the
  window centre (`reference_normalised_target`)
- SHA-256: `92625e7a9212d56756ccf6dcb374f4ae4bc0fb6cc01e422b258e4709f07fb8d4`

The cache contains opaque participant and trial labels, side-specific channels,
modelling features, targets, and QC/LOSO fields only. It contains no raw EMG
samples, identifying information, or mapping to source-study labels. See
[Model reproduction](../../docs/model-reproduction.md) for the validation and
rebuild workflow.
