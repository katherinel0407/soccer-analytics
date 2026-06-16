import streamlit as st
import pandas as pd

from match_predictor import predict_match
from player_predictor import predict_player_goals

st.title("⚽ Soccer Analytics App")

team_df = pd.read_csv("../data/team_ratings.csv")
player_df = pd.read_csv("../data/player_ratings.csv")

teams = sorted(team_df["team"].unique())
players = sorted(player_df["player"].unique())

# tabs
tab1, tab2 = st.tabs(["Match Predictor", "Player Predictor"])

# for tab 1:
with tab1:
    st.header("Match Outcome Predictor")

    home_team = st.selectbox(
        "Home Team",
        teams
    )

    away_team = st.selectbox(
        "Away Team",
        teams,
        index=1
    )

    if st.button("Predict Match Outcome"):

        result = predict_match(
            home_team,
            away_team
        )

        st.subheader("Results")

        st.write(
            f"Expected Goals: "
            f"{result['home_xg']:.2f} - "
            f"{result['away_xg']:.2f}"
        )

        st.write(
            f"Home Win: {result['home_win']:.2%}"
        )

        st.write(
            f"Draw: {result['draw']:.2%}"
        )

        st.write(
            f"Away Win: {result['away_win']:.2%}"
        )

# tab 2:
with tab2:
    st.header("Player Goal Predictor")

    player = st.selectbox(
        "Player",
        players
    )

    if st.button("Predict Goals"):

        goals = predict_player_goals(player)

        st.subheader("Prediction")

        st.write(
            f"Expected Goals per Match: "
            f"{goals:.2f}"
        )