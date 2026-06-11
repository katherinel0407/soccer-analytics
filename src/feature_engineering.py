import numpy as np

# functions for variable calculations
# distance (dataset uses 120 x 40 unit system)
def calc_distance(x, y):
    return np.sqrt(
        (120 - x)**2 +
        (40 - y)**2
    )

# doing some angle calculations
def calc_angle(x, y):
    goal_width = 7.32

    numerator = goal_width * x
    denominator = (x**2 + y**2 - (goal_width**2 / 4))

    return np.arctan2(numerator, denominator)

# what body part was used
def extract_body_part(shot):
    return shot.get(
        "body_part",
        {}
    ).get(
        "name",
        "Unknown"
    )

# what type of shot
def extract_shot_type(shot):
    return shot.get(
        "type",
        {}
    ).get(
        "name",
        "Unknown"
    )


# to calculate number of opponents at time of shot ("freeze_frame") within a 5 "unit" radius
def count_nearby_opponents(
    shot_location,
    freeze_frame,
    radius=5
):
    # if info doesn't exist
    if not freeze_frame:
        return 0

    shot_x, shot_y = shot_location

    count = 0

    for player in freeze_frame:

        if player["teammate"]:
            continue

        player_x, player_y = player["location"]

        distance = np.sqrt(
            (shot_x - player_x)**2 +
            (shot_y - player_y)**2
        )

        if distance <= radius:
            count += 1

    return count

# is this player's first time shooting the ball?
def extract_first_time(shot):
    return int(
        shot.get("first_time", False)
    )

# is the player under pressure?
def extract_under_pressure(value):
    return int(bool(value))

# what technique was used to shoot?
def extract_technique(shot):
    return shot.get(
        "technique",
        {}
    ).get(
        "name",
        "Unknown"
    )

# building final dataset
def build_shot_dataset(event_df):

    shots = event_df[
        event_df["type"].apply(
            lambda x: x["name"] == "Shot"
        )
    ].copy()

    shots["goal"] = shots["shot"].apply(
        lambda x:
        x["outcome"]["name"] == "Goal"
    )

    shots["x"] = shots["location"].apply(
        lambda x: x[0]
    )

    shots["y"] = shots["location"].apply(
        lambda x: x[1]
    )

    shots["distance"] = shots.apply(
        lambda r:
        calc_distance(
            r["x"],
            r["y"]
        ),
        axis=1
    )

    shots["angle"] = shots.apply(
        lambda r:
        calc_angle(
            r["x"],
            r["y"]
        ),
        axis=1
    )

    shots["body_part"] = shots["shot"].apply(
        extract_body_part
    )

    shots["shot_type"] = shots["shot"].apply(
        extract_shot_type
    )
    
    shots["first_time"] = shots["shot"].apply(
        extract_first_time
    )

    shots["technique"] = shots["shot"].apply(
        extract_technique
    )

    shots["under_pressure"] = (
        shots["under_pressure"]
        .fillna(False)
        .astype(int)
    )


    shots["num_opponents"] = shots.apply(
        lambda row:
            count_nearby_opponents(
                row["location"],
                row["shot"].get("freeze_frame", []),
                radius=5
            ),
        axis=1
    )

    final_dataset = shots[
        [
            "distance",
            "angle",
            "goal",
            "body_part",
            "shot_type",
            "first_time",
            "under_pressure",
            "num_opponents",
            "technique"
        ]
    ]

    return final_dataset