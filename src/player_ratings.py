import pandas as pd

def build_player_ratings():

    df = pd.read_csv("../data/shot_predictions.csv")

    player_stats = df.groupby(["player", "team"]).agg(
        xG=("xG", "sum"),
        goals=("goal", "sum"),
        shots=("goal", "count")
    ).reset_index()

    player_stats["xG_per_shot"] = player_stats["xG"] / player_stats["shots"]
    player_stats["finishing_diff"] = player_stats["goals"] - player_stats["xG"]

    # estimate shooting rate (very simple baseline)
    player_stats["shots_per_match"] = player_stats["shots"] / 10  # approx scaling

    player_stats.to_csv("../data/player_ratings.csv", index=False)

if __name__ == "__main__":
    build_player_ratings()