import argparse

import pandas as pd

from emg_normalisation.cli.main import _validate_data
from emg_normalisation.io import write_emg_mot


def test_validate_data_checks_bundled_public_files(tmp_path):
    data_root = tmp_path / "data"
    manifests = data_root / "manifests"
    manifests.mkdir(parents=True)
    pd.DataFrame({"participant": ["C01"]}).to_csv(manifests / "trial_manifest.csv", index=False)
    derived = data_root / "derived"
    derived.mkdir()
    pd.DataFrame(
        {
            "participant": ["C01"],
            "trial": ["Cycling01"],
            "channel": ["r_vaslat"],
            "variance_mV2": [0.1],
            "kde_fwhm_mV": [0.2],
            "reference_normalised_target": [0.3],
        }
    ).to_parquet(derived / "window_features.parquet", index=False)
    example_root = tmp_path / "examples" / "C06" / "trial_ffc10db23f"
    write_emg_mot(
        pd.DataFrame({"time": [0.0, 0.001], "r_vaslat": [0.1, 0.2]}),
        example_root / "emgBPF.mot",
    )
    pd.DataFrame({"time": [0.0, 0.001], "r_vaslat": [0.1, 0.2]}).to_parquet(
        example_root / "reference_normalised_envelope.parquet", index=False
    )

    status = _validate_data(argparse.Namespace(data_root=data_root, example_root=example_root))

    assert status == 0
