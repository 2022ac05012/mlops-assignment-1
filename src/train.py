# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import mlflow
import mlflow.sklearn

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
mlflow.set_experiment("Random forest car evaluation")
with mlflow.start_run():
# Train a model
    n_estimators = 95
    random_state = 40
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
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

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1score", f1Score)
    mlflow.sklearn.log_model(model, "rf_default_for_cars")
    # Save the trained model
    joblib.dump(model, 'model/model.joblib')
    print("Model saved as 'model.joblib'")
