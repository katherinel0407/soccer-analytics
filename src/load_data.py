# loading data that can be used in the XGBoost model
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. combine all processed csv files into one large csv

processed_dir = Path("../data/processed")

csv_files = processed_dir.glob("shots_*.csv")

dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)


# observation/note: many of the csv files are missing information in some columns (eg. if there was no penalty taken, there is no penalty column in the entire csv). This is marked with a ,
# fill missing vals with 0

all_shots = pd.concat(dfs, ignore_index=True)

dummy_cols = [
    col for col in all_shots.columns
    if col not in ["distance", "angle", "goal"]
]

all_shots[dummy_cols] = (
    all_shots[dummy_cols]
    .fillna(False)
    .astype(int)
)

# change goal to 0/1 value
all_shots["goal"] = all_shots["goal"].astype(int)

all_shots.to_csv(
    "../data/processed/all_shots.csv",
    index=False
)

# 2. let's look a little more into the distributions
# all_shots.info()

# all_shots.isna().sum()

# print(all_shots["goal"].value_counts())

# # note: there's quite a data imbalance with 88% being no goal, and 11% resulting in a goal

# print(all_shots["distance"].describe())
# print(all_shots["angle"].describe())

# # good to have some plots!
# all_shots["distance"].hist(bins=50)
# plt.title("Shot Distance Distribution")
# plt.show()

# all_shots["angle"].hist(bins=50)
# plt.title("Shot Angle Distribution")
# plt.show()
