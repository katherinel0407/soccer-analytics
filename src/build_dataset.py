from pathlib import Path
import pandas as pd
from feature_engineering import build_shot_dataset

raw_dir = Path("../data/raw")
output_dir = Path("../data/processed")

for file_path in raw_dir.glob("*.json"):

    event = pd.read_json(file_path)

    shots = build_shot_dataset(event)

    shots.to_csv(
        output_dir /
        f"shots_{file_path.stem}.csv",
        index=False
    )