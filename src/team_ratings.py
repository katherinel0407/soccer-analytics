import pandas as pd

# this is to help with match predictions (strength rankings)
def build_team_ratings():

    df = pd.read_csv("../data/shot_predictions.csv")

    team_stats = df.groupby("team").agg(
        xG_for=("xG", "sum"),
        goals_for=("goal", "sum"),
        shots=("goal", "count")
    ).reset_index()

    team_stats["xG_per_shot"] = team_stats["xG_for"] / team_stats["shots"]
    team_stats["finishing_diff"] = team_stats["goals_for"] - team_stats["xG_for"]

    # normalize strength
    team_stats["attack_strength"] = team_stats["xG_for"] / team_stats["xG_for"].mean()

    team_stats.to_csv("../data/team_ratings.csv", index=False)

if __name__ == "__main__":
    build_team_ratings()