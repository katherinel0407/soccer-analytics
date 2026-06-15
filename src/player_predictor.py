import pandas as pd

def predict_player_goals(player_name):

    df = pd.read_csv("../data/player_ratings.csv")

    player = df[df["player"] == player_name]

    if player.empty:
        raise ValueError(f"Player {player_name} not found in dataset")

    player = player.iloc[0]

    # expected shots per match (baseline model)
    expected_shots = player["shots_per_match"]

    # expected xG per shot
    xg_per_shot = player["xG_per_shot"]

    expected_goals = expected_shots * xg_per_shot

    print(f"\nPlayer: {player_name}")
    print(f"Expected Goals per Match: {expected_goals:.2f}")
    print(f"Expected Shots: {expected_shots:.2f}")
    print(f"xG per Shot: {xg_per_shot:.2f}")

    return expected_goals


if __name__ == "__main__":
    player = input("Enter Player Name: ")

    predict_player_goals(player)