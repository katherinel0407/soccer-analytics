# Soccer Analytics: Expected Goals (xG) Prediction System

## Overview

This project builds an Expected Goals (xG) model using StatsBomb event data and XGBoost to estimate the probability that a soccer shot results in a goal.

The system processes raw match event data, engineers shot-level features, trains a machine learning model, and generates player and team-level analytics based on predicted xG values.

The goal of the project is to demonstrate an end-to-end machine learning pipeline including:

- Data processing
- Feature engineering
- Predictive modeling
- Model evaluation
- Sports analytics applications

---

## Dataset

Data source:

https://github.com/statsbomb/open-data

![StatsBomb logo](img/SB_logo.png)

There are 4235 matches/events included in this dataset. The dataset contains detailed event-level information, including:

- Shot locations
- Shot outcomes
- Body part used
- Shot type
- Technique
- Defensive positioning
- Freeze-frame player locations
- Pressure information

---

## Project Structure

```text
soccer-analytics/
│
├── data/
│   ├── processed/
│   ├── shot_predictions.csv
│   ├── player_xg.csv
│   ├── team_xg.csv
│   ├── team_ratings.csv
│   └── matches.csv
│
├── models/
│   ├── xgboost_xg_model.pkl
│   └── features.pkl
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── feature_engineering.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── build_dataset.py
│   ├── load_data.py
│   ├── train_model.py
│   ├── xg_tables.py
│   ├── team_ratings.py
│   ├── match_dataset.py
│   ├── match_predictor.py
│   └── run_match_predictor.py
|
├── img/
│   ├── player_xg_ex.png
│   ├── SB_logo.png
│   └── team_xg_ex.png
│
└── README.md
```

---

## Feature Engineering

For each shot (108281 in total across all 4235 events), the following features are extracted:

### Shot Geometry

- Distance from goal
- Shooting angle

### Shot Characteristics

#### Body Part

- Left Foot
- Right Foot
- Head
- Other

#### Shot Type

- Open Play
- Penalty
- Free Kick
- Corner

#### Technique

- Normal
- Volley
- Half Volley
- Backheel
- Lob
- Other

### Contextual Features

- First-time shot indicator
- Under-pressure indicator
- Number of nearby defenders within a 5 unit radius (freeze-frame data)

### Target Variable

```text
goal = 1
not goal = 0
```

---

## Model

The project uses XGBoost because of its strong performance on tabular datasets and ability to model nonlinear relationships between shot characteristics. Prior to using this model, preliminary testing included usage of a logisitic regression, but was later changed due to reasons stated above. With changes to the XGBoost model, accuracy increased from 0.78 to 0.80. 

### Hyperparameters

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

---

## Model Performance

| Metric | Value |
|----------|----------|
| ROC-AUC | 0.80 |

---

## Outputs

### Shot-Level Predictions

`shot_predictions.csv`

Contains:

- Match ID
- Team
- Player
- Actual outcome
- Shot characteristics
- Predicted xG

### Player Analytics

`player_xg.csv`

Contains:

- Player name
- Player team
- Predicted xG
- Goals scored
- Total shots
- Goals minus xG

Used to identify:

- High performing finishers
- Low performing finishers
- High-volume shooters

Example: Top 10 players by xG
![Top 10 players](img/player_xg_ex.png)

### Team Analytics

`team_xg.csv`

Contains:

- Team name
- Team xG
- Team goals
- Team shots
- Goals minus xG

Used to evaluate:

- Attacking quality
- Finishing efficiency
- Team performance trends

Example: Top 10 teams by xG
![Top 10 teams](img/team_xg_ex.png)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook

---

## How to Run

### Feature Engineering

```bash
python src/feature_engineering.py
```

### Building Dataset for Model

```bash
python src/build_dataset.py
```

### Loading Data 

```bash
python src/load_data.py
```

### Training Model

```bash
python/train_model.py
```

### Create Player and Team Predictions
```bash
python/xg_tables.py
```

---

## Match Prediction (Interactive CLI Tool)

The project includes an interactive command-line tool that allows users to input any two teams and receive match predictions powered by the trained xG model and team strength ratings.

### Running the Predictor

```bash
python src/run_match_predictor.py
```

You will be prompted to enter:

```text
Enter home team: [your choice of home team]
Enter away team: [your choice of away team]
```

---

## How It Works

1. Loads precomputed team strength ratings from `team_ratings.csv`
2. Retrieves attack strength for both teams
3. Estimates expected goals using learned team strengths
4. Simulates match outcomes using a Poisson model
5. Home team has an added 30% boost to account for home stadium advantage
6. Outputs:
   - Expected goals (xG) for both teams
   - Win / Draw / Loss probabilities

---

## Future Improvements

### Additional Features

- Goalkeeper position
- Nearest defender
- Match state
- Timestamp

### Model Parameters

- Hyperparamter tuning
- Cross-validation

### Visualization

- Potential interactive streamlit dashboard
- Player comparison dashboards

### Team Naming

Because of how StatsBomb data is created, there is a slight issue with how teams are named. For a few teams, they are not titled by their official name, rather the name that fans are more accustomed to saying (e.g. Barcelona vs. FC Barcelona). In addition, women's teams naming can vary from [team name] Women’s, [team name] W, or [team name] WFC. Improvements can be made by creating uniform team names and an extra validation step when user is entering their home and away teams of choice.

---

## Author

**Katherine Li**