from pathlib import Path
import pandas as pd

def load_raw_data():
    dfs = []
    for file in Path("../data/processed").glob("shots_*.csv"):
        dfs.append(pd.read_csv(file))
    return pd.concat(dfs, ignore_index=True)

def load_model_data():
    df = load_raw_data()
    df = pd.get_dummies(df, columns=["body_part", "shot_type", "technique"], dtype=int)
    return df