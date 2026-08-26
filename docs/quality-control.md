# Quality control

## Channel exclusions

Five channels were excluded completely after visual inspection identified
unreliable adductor recordings:

- C01 right ADD;
- C03 right ADD;
- C04 right ADD;
- C05 right ADD;
- C07 left ADD.

They are absent from the reference targets, bundled feature table, training,
predictions, metrics, and figures.

## Reference overrides

- E01 right TA uses `trial_4d068ec7e2`.
- C05 right GMax uses `trial_b4af0473c9`.
- C06 right ADD uses `trial_49b3331954`.

Overrides replace a visually rejected or protocol-specific automatic maximum.

## Trial roles

- Seventeen cycling conditions are primary cycling outcomes.
- Three cycling-cohort CMJs per participant augment model training and reference
  normalisation but are not primary cycling conditions.
- E01 is never used to train the cycling model.
- All E01 trials can contribute to its reference normalisation; task groups are
  retained only as analysis roles.

## Cadence mapping

The condition manifest uses opaque trial and condition labels. It retains the
analysis mapping required for aggregation without retaining source-study labels.
