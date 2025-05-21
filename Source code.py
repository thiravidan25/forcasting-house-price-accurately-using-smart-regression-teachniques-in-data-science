# === Upload and Load Data ===
from google.colab import files
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Prompt user to upload the CSV file
uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding="latin-1")

# === Preprocessing ===
df = df.drop_duplicates()
df = df.dropna()
df.columns = df.columns.str.lower()
df.reset_index(drop=True, inplace=True)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Exploratory Data Analysis ===
# 1. Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 2. Distribution of Target Variable (Price)
plt.figure(figsize=(10, 5))
sns.histplot(df['price'], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# === Features and Target ===
X = df.drop("price", axis=1)
y = df["price"]

# === Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === Model 1: Linear Regression ===
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("=== Linear Regression Results ===")
print(f"MAE:  {mean_absolute_error(y_test, lr_pred):.2f}")
print(f"MSE:  {mean_squared_error(y_test, lr_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")
print(f"R²:   {r2_score(y_test, lr_pred):.2f}\n")

# === Model 2: Random Forest Regressor ===
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("=== Random Forest Regressor Results ===")
print(f"MAE:  {mean_absolute_error(y_test, rf_pred):.2f}")
print(f"MSE:  {mean_squared_error(y_test, rf_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")
print(f"R²:   {r2_score(y_test, rf_pred):.2f}\n")

# === Feature Importance from Random Forest ===
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importance.sort_values(ascending=False).head(15).plot(kind='bar')
plt.title("Top 15 Feature Importances")
plt.ylabel("Importance Score")
plt.show()

# === Predict Prices for 5 Sample Inputs ===
sample_inputs = X_test[:5]
sample_preds = rf_model.predict(sample_inputs)

print("=== Predicted Prices for 5 Sample Inputs ===")
for idx, price in enumerate(sample_preds, 1):
    print(f"Input {idx}: ₹{price:,.2f}")
