from match_predictor import predict_match

# allows for user input of what two teams to predict match outcome
def main():

    print("\n⚽ Soccer Match Predictor ⚽\n")

    home_team = input("Enter home team: ")
    away_team = input("Enter away team: ")

    print("\nPredicting match...\n")

    predict_match(home_team, away_team)


if __name__ == "__main__":
    main()