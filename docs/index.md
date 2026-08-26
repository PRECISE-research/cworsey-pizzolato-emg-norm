# EMG normalisation documentation

This package implements the distribution-informed EMG normalisation workflow
reported by Carrall-Worsey and Pizzolato.

The repository has three complementary uses:

1. apply the published cycling-cohort models to compatible bandpass EMG;
2. repeat C01-C07 LOSO validation and rebuild the published linear models from
   the bundled deidentified cohort feature table;
3. train alternative estimators using the same data, folds, reconstruction, and
   evaluation definitions.

The manuscript method predicts a normalised trial peak from local amplitude
distributions and uses that peak to scale the conventional linear envelope.
The package can also return the interpolated model trace directly.

Begin with [Getting started](getting-started.md).
