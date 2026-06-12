import joblib
import pandas as pd
from load_data import load_raw_data
import matplotlib.pyplot as plt

model = joblib.load("../models/xgboost_xg_model.pkl")

all_shots = load_raw_data()

model_input = pd.get_dummies(
    all_shots,
    columns=["body_part", "shot_type", "technique"],
    dtype=int
)

# defining our feature columns
features = joblib.load("../models/features.pkl")
X = model_input[features]

# adding xG values
all_shots["xG"] = model.predict_proba(X)[:, 1]
all_shots.to_csv("../data/shot_predictions.csv", index=False)

# now let's build player xG tables!
df = pd.read_csv("../data/shot_predictions.csv")

player_xg = df.groupby(["player", "team"]).agg(
    xG=("xG", "sum"),
    goals=("goal", "sum"),
    shots=("goal", "count")
).reset_index()

player_xg["xG_diff"] = player_xg["goals"] - player_xg["xG"]

player_xg.to_csv("../data/player_xg.csv", index=False)

# let's visualize this: who has highest player xG?
top_players = player_xg.sort_values("xG", ascending=False).head(10)

plt.barh(top_players["player"], top_players["xG"])
plt.title("Top Players by xG")
plt.show()

# let's do the same for teams
team_xg = df.groupby("team").agg(
    xG=("xG", "sum"),
    goals=("goal", "sum"),
    shots=("goal", "count")
).reset_index()

team_xg["xG_diff"] = team_xg["goals"] - team_xg["xG"]

team_xg.to_csv("../data/team_xg.csv", index=False)

top_teams = team_xg.sort_values("xG", ascending=False).head(10)

plt.barh(top_teams["team"], top_teams["xG"])
plt.title("Top Teams by xG")
plt.show()