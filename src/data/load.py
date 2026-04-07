from pathlib import Path

import pandas as pd


def load_csv(path, columns=None):

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichero no encontrado: {path}\n"
            "Verifica que los datos de MIMIC-IV están en data/raw/mimic_iv/hosp/"
        )

    return pd.read_csv(
        path,
        usecols=columns,
        low_memory=False
    )


def load_multiple_datasets(file_dict):

    datasets = {}

    for name, path in file_dict.items():
        print(f"Loading {name}...")
        datasets[name] = load_csv(path)

    return datasets