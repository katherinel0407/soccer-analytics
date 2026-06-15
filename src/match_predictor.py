import pandas as pd
import numpy as np

def load_team_ratings():

    return pd.read_csv("../data/team_ratings.csv")


# we will use a poisson simulation to predict outcome
def simulate_match(home_xg, away_xg, n_sim=10000):

    home_goals = np.random.poisson(home_xg, n_sim)
    away_goals = np.random.poisson(away_xg, n_sim)

    home_win = np.mean(home_goals > away_goals)
    draw = np.mean(home_goals == away_goals)
    away_win = np.mean(home_goals < away_goals)

    return home_win, draw, away_win

def predict_match(home_team, away_team):

    ratings = load_team_ratings()

    home = ratings[ratings["team"] == home_team].iloc[0]
    away = ratings[ratings["team"] == away_team].iloc[0]

    # expected goals (simple strength model), allows for slight advantage to home team
    home_xg = home["attack_strength"] * 1.5
    away_xg = away["attack_strength"] * 1.2

    home_win, draw, away_win = simulate_match(home_xg, away_xg)

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win
    }

    print(f"\n{home_team} vs {away_team}")
    print(f"Expected Goals: {home_xg:.2f} - {away_xg:.2f}")
    print(f"Home Win: {home_win:.2%}")
    print(f"Draw: {draw:.2%}")
    print(f"Away Win: {away_win:.2%}")

    return result


if __name__ == "__main__":

    home_team = input("Enter home team: ")
    away_team = input("Enter away team: ")

    predict_match(home_team, away_team)