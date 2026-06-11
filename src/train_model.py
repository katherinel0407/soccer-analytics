# this is where we train our model

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from load_data import load_all_shots

from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

all_shots = load_all_shots()

# x has all our features
X = all_shots.drop(columns=["goal"])
# y is the result (goal: 0/1)
y = all_shots["goal"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# train model: XGBoost Model
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)

# generate XG predictions
xg_pred = xgb.predict_proba(X_test)[:, 1]

# saving the model
joblib.dump(
    xgb,
    "../models/xgboost_xg_model.pkl"
)

# now we look at model accuracy and feature importance to get a sense of our results
auc = roc_auc_score(
    y_test,
    xg_pred
)

print(auc) # we get a value of 0.8014

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb.feature_importances_
})

importance.sort_values(
    by="importance",
    ascending=True
)

print(importance)

# map importance
plt.figure(figsize=(8,6))
plt.barh(
    importance["feature"],
    importance["importance"]
)
plt.title("XGBoost Feature Importance")
plt.show()