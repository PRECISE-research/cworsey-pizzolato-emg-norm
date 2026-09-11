# Security and responsible disclosure

## Scope

This repository publishes research software and deidentified human-derived
research data. Two kinds of report are in scope:

1. **Software vulnerabilities** in the `emg_normalisation` package, its command
   line interface, or the example scripts.
2. **Data-protection concerns**, including any suspicion that the published
   feature cache, example trial, or manifests could contribute to
   reidentification of a participant.

## Reporting

Report software vulnerabilities through GitHub's private reporting workflow:
**Security -> Report a vulnerability** on this repository. Private reports are
visible only to the maintainers.

Report a data-protection or reidentification concern the same way, and mark it
clearly as a data concern so it is triaged first. **Do not open a public issue
for a data concern, and do not include any identifying information in your
report.**

Please allow 30 days for an initial assessment. This is academic research
software maintained alongside other work, so response times are best effort
rather than guaranteed.

## Handling

Confirmed data-protection concerns are escalated to the Griffith University
ethics and research-governance contacts responsible for approvals 2023/444 and
2022/762 before any public statement is made. Affected data may be withdrawn
from the repository while that review is underway.

## Supported versions

Fixes are applied to the `main` branch. Published model artefacts are versioned
and are not rewritten in place; a correction is released as a new version with
an updated `PROVENANCE.md`.
