# hyperparameter_tuning.py
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.2f}")
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")   
precision = precision_score(y_test, y_pred, average='micro')
print(f"Precision Score: {precision:.2f}")
# Calculate Mean Absolute Error (MAE)
recall = recall_score(y_test, y_pred, average='micro')
print(f"Recall score: {recall:.2f}")

# Calculate Mean Squared Error (MSE)
f1Score = f1_score(y_test, y_pred, average='micro')
print(f"F1 score: {f1Score:.2f}")

# Calculate Root Mean Squared Error (RMSE)
report = classification_report(y_test, y_pred)

# Save the best model

joblib.dump(best_model, 'model/best_model.joblib')
print("Best model saved as 'best_model.joblib'")
