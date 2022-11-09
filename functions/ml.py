import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
import pickle


def train_test_split(train, test):
    X_train = train.iloc[:, 2:]
    X_test = test.iloc[:, 2:]
    y_train = train.iloc[:, 1]
    y_test = test.iloc[:, 1]
    #X_train = X_train.iloc[:, :14]
    #X_test = X_test.iloc[:, :14]

    return X_train, X_test, y_train, y_test


def build_model():
    model = XGBRegressor(n_estimators=100000, learning_rate=0.01, early_stopping_rounds=100)
    cv = KFold(n_splits=3)
    params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 10, 14]
    }
    clf = GridSearchCV(estimator=model, param_grid=params, cv=cv)
    return clf


def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)


def plot_diagnostics(y_test, y_pred):
    print(f'MSE: {round(mean_squared_error(y_true=y_test, y_pred=y_pred),2)}')
    print(f'RMSE: {round(np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)),2)}')
    print(f'MAE: {round(mean_absolute_error(y_true=y_test, y_pred=y_pred),2)}')


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return y_pred


def save_model(model, filepath):
    return pickle.dump(model, open(filepath, 'wb'))


def load_model(filepath):
    return pickle.load(open(filepath, 'rb'))
