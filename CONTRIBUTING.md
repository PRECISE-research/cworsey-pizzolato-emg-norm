# Contributing

Contributions that improve reproducibility, documentation, tests, or
model-benchmark interfaces are welcome.

By taking part you agree to the [code of conduct](CODE_OF_CONDUCT.md).

1. Create a branch from `main`.
2. Install development dependencies with `pip install -e ".[dev]"`.
3. Add tests for behavioural changes.
4. Run `pytest`, `ruff check src tests examples`, and
   `mkdocs build --strict`.
5. Do not add participant identifiers, unapproved human-derived data, Word
   drafts, or local absolute paths.
6. Keep alternative-model outputs separate from the published model bundle.

Changes to trial roles, reference maxima, cadence mappings, or channel
exclusions require scientific justification and an updated audit manifest.
