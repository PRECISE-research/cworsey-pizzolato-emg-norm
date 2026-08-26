# Published cycling-cohort models

This directory contains one multiple linear least-squares model for each of 13
anatomical muscles. Left and right channels were pooled during fitting but are
predicted separately.

For window \(k\):

\[
\hat{y}_k = \beta_0 + \beta_v z(v_k) + \beta_\omega z(\omega_k)
\]

where \(v_k\) is variance in mV², \(\omega_k\) is KDE FWHM in mV, and each
feature is standardised using the muscle-specific training mean and standard
deviation in `coefficients.csv`.

- `model.json`: transparent model definitions used by the Python API.
- `coefficients.csv`: human-readable equations and standardisation values.
- `model.pkl`: optional exact Python serialisation of the public model objects.
- `model_card.json`: intended use, provenance, and limitations.

The models were fitted to the seven cycling participants, including their
cycling and countermovement-jump training trials. `E01` was not used to
fit these models.
