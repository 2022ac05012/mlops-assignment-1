# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
data = pd.read_csv('data/car_evaluation.csv', header=None);
data[0].replace(['vhigh', 'high','med','low'], [3, 2, 1, 0], inplace=True)
data[1].replace(['vhigh', 'high','med','low'], [3, 2, 1, 0], inplace=True)
data[2].replace(['5more'], [5], inplace=True)
data[3].replace(['more'], [1], inplace=True)
data[4].replace(['big','med','small'], [2,1,0], inplace=True)
data[5].replace(['high','med','low'], [2,1,0], inplace=True)
data[6].replace(['vgood','good','acc','unacc'], [3,2,1,0], inplace=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.2f}")

# Save the trained model
joblib.dump(model, 'model/model.joblib')
print("Model saved as 'model.joblib'")
