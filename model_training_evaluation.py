import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load processed data
df = pd.read_csv('hyundi_processed.csv')

# 2. Split into features and target, then train/test split

X_train, X_test = train_test_split(df, test_size=0.2, random_state=0)

# 3. Train AutoGluon and save model directory
gl_model_dir = "ag_model_dir"
predictor = TabularPredictor(label="price", problem_type = 'regression', eval_metric = 'mean_squared_error', path=gl_model_dir).fit(train_data = X_train, presets = "best_quality")



# 4. Generate predictions and compute metrics

X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

preds = predictor.predict(X_test)
metrics = {
    "MAE": mean_absolute_error(y_test, preds),
    "MSE": mean_squared_error(y_test, preds),
    "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
    "R2": r2_score(y_test, preds)
}

# 5. Print and save metrics to JSON
print(json.dumps(metrics, indent=2))
with open("metrics.json", "w") as mf:
    json.dump(metrics, mf, indent=2)

# 6. Report model directory location
print(f"AutoGluon model saved to: {gl_model_dir}")
