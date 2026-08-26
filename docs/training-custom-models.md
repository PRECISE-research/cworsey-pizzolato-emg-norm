# Training alternative models

Alternative estimators use exactly the same real, QC-filtered C01-C07 features,
reference targets, and fixed LOSO splits as the published model. Download the
bundled cache with Git LFS before training:

```bash
git lfs pull --include="data/derived/window_features.parquet"
```

## Try a built-in model

No Python model code is required for the built-in estimators:

```bash
emg-normalisation train \
  --features data/derived/window_features.parquet \
  --model ridge \
  --output-dir outputs/ridge

emg-normalisation train \
  --features data/derived/window_features.parquet \
  --model hist-gradient-boosting \
  --output-dir outputs/hist-gradient-boosting

emg-normalisation train \
  --features data/derived/window_features.parquet \
  --model random-forest \
  --output-dir outputs/random-forest
```

| Name | Model family | Useful as |
|---|---|---|
| `linear` | Standardised ordinary least squares | Published baseline |
| `ridge` | Scaled L2-regularised linear regression | Regularised linear comparison |
| `hist-gradient-boosting` | Histogram gradient-boosted trees | Scalable nonlinear comparison |
| `random-forest` | Random-forest regression | Flexible ensemble comparison |

The nonlinear defaults are deterministic and intentionally conservative. They
are convenient starting points, not data-tuned recommendations. Each command
writes held-out window predictions, fold-by-muscle metrics, and fitted fold
models to its own output directory.

## Bring your own model

For an estimator not in the built-in catalogue, provide a factory with the
following interface.

```python
class MyModel:
    def fit(self, features, targets):
        return self

    def predict(self, features):
        return predictions

    def save(self, path):
        ...
```

Use `SklearnEstimatorAdapter` for scikit-learn-compatible models:

```python
from sklearn.ensemble import RandomForestRegressor
from emg_normalisation.models import SklearnEstimatorAdapter

def create_model():
    estimator = RandomForestRegressor(
        n_estimators=200,
        random_state=1,
        n_jobs=-1,
    )
    return SklearnEstimatorAdapter(estimator)
```

Run the custom factory:

```bash
emg-normalisation train \
  --features data/derived/window_features.parquet \
  --model examples/custom_model.py:create_model \
  --output-dir outputs/random_forest
```

The pipeline fits a separate estimator for each anatomical muscle in each LOSO
fold. A held-out participant is never used for fitting or model-specific
preprocessing. Predictions are bounded to 0-1 after the estimator returns.

## Fair comparisons

Use the supplied fold and QC manifests. Report cycle-level absolute peak error
as the primary endpoint. Window-level MAE, RMSE, and R2 are diagnostic and
should not replace the cycle-level manuscript endpoint.

If a model has its own feature scaling or tuning, fit it using training rows
only. Comparing the supplied defaults does not establish which model is best.
Nested cross-validation is required when hyperparameters or a winning model
family are selected from data.

## PyTorch-style models

Deep-learning frameworks are intentionally not dependencies. A wrapper only
needs to convert the NumPy arrays received by `fit` and `predict`:

```python
class TorchRegressor:
    def __init__(self):
        import torch
        self.torch = torch
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

    def fit(self, features, targets):
        x = self.torch.as_tensor(features, dtype=self.torch.float32)
        y = self.torch.as_tensor(targets[:, None], dtype=self.torch.float32)
        optimiser = self.torch.optim.Adam(self.network.parameters(), lr=1e-3)
        for _ in range(100):
            loss = ((self.network(x) - y) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return self

    def predict(self, features):
        x = self.torch.as_tensor(features, dtype=self.torch.float32)
        with self.torch.no_grad():
            return self.network(x).numpy().ravel()

    def save(self, path):
        self.torch.save(self.network.state_dict(), path)
        return path
```

This is an interface example, not a recommended architecture. Model selection,
early stopping, and all learned preprocessing must remain inside the training
participants of each outer LOSO fold.
