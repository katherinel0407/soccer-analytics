from pathlib import Path
import pandas as pd

def load_all_shots():

    processed_dir = Path("../data/processed")

    csv_files = processed_dir.glob("shots_*.csv")

    dfs = []

    for file in csv_files:
        dfs.append(pd.read_csv(file))

    all_shots = pd.concat(dfs, ignore_index=True)

    all_shots = pd.get_dummies(
        all_shots,
        columns=[
            "body_part",
            "shot_type",
            "technique"
        ],
        dtype=int
    )

    all_shots["goal"] = all_shots["goal"].astype(int)

    return all_shots