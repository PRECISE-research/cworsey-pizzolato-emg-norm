# Data directory

This checkout includes the deidentified C01-C07 feature and target cache at
`data/derived/window_features.parquet`. It does not include all-trial raw EMG,
reference envelopes, or segmentation files. A
single anonymised real-data example is kept under `examples/full_trial_data/`.

The retained files under `data/manifests/` document the public trial labels,
roles, channel metadata, QC exclusions, checksums, and reference-normalisation
metadata used by the manuscript workflow. They are retained as concise metadata
for transparency. The bundled feature table is the complete public input for
LOSO validation and rebuilding the linear models.
