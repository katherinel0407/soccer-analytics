# building dataset for match predictions --> match-level data for shots
import pandas as pd

def build_match_dataset():

    df = pd.read_csv("../data/shot_predictions.csv")

    # aggregate per match + team
    match_team = df.groupby(["match_id", "team"]).agg(
        xG=("xG", "sum"),
        goals=("goal", "sum"),
        shots=("goal", "count")
    ).reset_index()

    # pivot into home/away style structure
    matches = match_team.pivot(
        index="match_id",
        columns="team",
        values=["xG", "goals", "shots"]
    )

    # flatten columns
    matches.columns = ["_".join(col).strip() for col in matches.columns]
    matches = matches.reset_index()

    # replace NaNs with 0
    matches = matches.fillna(0)

    matches.to_csv("../data/matches.csv", index=False)

if __name__ == "__main__":
    build_match_dataset()