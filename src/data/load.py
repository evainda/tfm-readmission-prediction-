import pandas as pd


def load_csv(path, columns=None):

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