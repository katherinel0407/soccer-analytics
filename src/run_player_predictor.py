from player_predictor import predict_player_goals

def main():

    print("\n⚽ Player Goal Predictor ⚽\n")

    player_name = input("Enter player name: ")

    print("\nPredicting...\n")

    predict_player_goals(player_name)

# if __name__ == "__main__":
#     main()