import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import joblib
import os

# Test data loading and preprocessing
def test_data_loading_and_preprocessing():
    data = pd.read_csv('data/car_evaluation.csv', header=None)
    assert not data.empty, "Dataset is empty"
    assert data.shape[1] == 7, "Dataset does not have the expected number of columns"

    # Preprocessing
    data[0].replace(['vhigh', 'high', 'med', 'low'], [3, 2, 1, 0], inplace=True)
    data[1].replace(['vhigh', 'high', 'med', 'low'], [3, 2, 1, 0], inplace=True)
    data[2].replace(['5more'], [5], inplace=True)
    data[3].replace(['more'], [1], inplace=True)
    data[4].replace(['big', 'med', 'small'], [2, 1, 0], inplace=True)
    data[5].replace(['high', 'med', 'low'], [2, 1, 0], inplace=True)
    data[6].replace(['vgood', 'good', 'acc', 'unacc'], [3, 2, 1, 0], inplace=True)

    assert data.isnull().sum().sum() == 0, "Dataset contains null values"
    assert set(data[6].unique()) == {0, 1, 2, 3}, "Unexpected class labels in the target column"

# Test train-test split
def test_train_test_split():
    data = pd.read_csv('data/car_evaluation.csv', header=None)
    data[0].replace(['vhigh', 'high', 'med', 'low'], [3, 2, 1, 0], inplace=True)
    data[1].replace(['vhigh', 'high', 'med', 'low'], [3, 2, 1, 0], inplace=True)
    data[2].replace(['5more'], [5], inplace=True)
    data[3].replace(['more'], [1], inplace=True)
    data[4].replace(['big', 'med', 'small'], [2, 1, 0], inplace=True)
    data[5].replace(['high', 'med', 'low'], [2, 1, 0], inplace=True)
    data[6].replace(['vgood', 'good', 'acc', 'unacc'], [3, 2, 1, 0], inplace=True)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) > 0, "Training set is empty"
    assert len(X_test) > 0, "Test set is empty"
    assert len(y_train) > 0, "Training labels are empty"
    assert len(y_test) > 0, "Test labels are empty"

# Test model training
def test_model_training():
    data = pd.read_csv('data/car_evaluation.csv', header=None)
    data[0].replace(['vhigh', 'high', 'med', 'low'], [3, 2, 1, 0], inplace=True)
    data[1].replace(['vhigh', 'high', 'med', 'low'], [3, 2, 1, 0], inplace=True)
    data[2].replace(['5more'], [5], inplace=True)
    data[3].replace(['more'], [1], inplace=True)
    data[4].replace(['big', 'med', 'small'], [2, 1, 0], inplace=True)
    data[5].replace(['high', 'med', 'low'], [2, 1, 0], inplace=True)
    data[6].replace(['vgood', 'good', 'acc', 'unacc'], [3, 2, 1, 0], inplace=True)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=95, random_state=40)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    assert accuracy > 0.7, "Model accuracy is too low"
    assert precision > 0.7, "Model precision is too low"
    assert recall > 0.7, "Model recall is too low"
    assert f1 > 0.7, "Model F1 score is too low"

# Test model saving
def test_model_saving():
    model = RandomForestClassifier(n_estimators=95, random_state=40)
    joblib.dump(model, 'model/test_model.joblib')

    assert os.path.exists('model/test_model.joblib'), "Model file was not saved"
    os.remove('model/test_model.joblib')  # Clean up after test

