import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import preprocess_dataframe


# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("insurance.csv")

# -----------------------------
# 2. Separate target variable
# -----------------------------
y = df["charges"]
X = df.drop("charges", axis=1)

# -----------------------------
# 3. Preprocess features
# -----------------------------
X = preprocess_dataframe(X)

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# -----------------------------
# 5. Train Linear Regression
# -----------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\nLinear Regression Results")
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)

# -----------------------------
# 6. Train Random Forest
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# -----------------------------
# 7. Save the best model
# -----------------------------
if rf_r2 > lr_r2:
    print("\nSaving Random Forest model")
    joblib.dump(rf_model, "model/health_insurance_model.pkl")
else:
    print("\nSaving Linear Regression model")
    joblib.dump(lr_model, "model/health_insurance_model.pkl")
