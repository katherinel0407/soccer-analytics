# this is where we train our model

from sklearn.model_selection import train_test_split
from load_data import all_shots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd

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

# train model: choice of logistic regression as a baseline model
lr = LogisticRegression(
    max_iter=1000
)

lr.fit(X_train, y_train)

# generate XG predictions
xg_pred = lr.predict_proba(X_test)[:, 1]

# now we look at model accuracy and feature importance to get a sense of our results
auc = roc_auc_score(
    y_test,
    xg_pred
)

print(auc) # we get a value of 0.7839755130197827

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": lr.coef_[0]
})

coef_df.sort_values(
    by="coefficient",
    ascending=False
)

print(coef_df)
