import pickle
import sys
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from functions import plot_diagnostics
from dataProcessing import DataProcessor
import pickle

def load_data(train_dataset_filepath, test_dataset_filepath, data_processor):
    df_train = data_processor.read_data(train_dataset_filepath)
    df_test = data_processor.read_data(test_dataset_filepath)
    
    df_train['index'] = df_train['index'].astype('int64')
    df_test['index'] = df_test['index'].astype('int64')

    return df_train, df_test

def train_test_split(training_data, testing_data):
    X_train = training_data.iloc[:, 4:]
    X_test = testing_data.iloc[:, 4:]

    y_train = training_data['venda']
    y_test = testing_data['venda']

    return X_train, X_test, y_train, y_test

def build_model():

    model = XGBRegressor(n_estimators=10000, learning_rate=0.001, early_stopping_rounds=50)

    params = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [6, 10, 12]
    }   

    cv = TimeSeriesSplit(n_splits=5)

    clf = GridSearchCV(estimator=model, param_grid=params, cv=cv)

    return clf

def evaluate_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def save_model(model, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 5:
        dt = DataProcessor()

        train_data, test_data, original_testing_dataset, modelpath= sys.argv[1:]
        print('Loading data...\n')

        df_train, df_test = load_data(train_data, test_data, dt)

        original_testing_dataset = pd.read_csv(original_testing_dataset)

        print('Building training and testing datasets...\n')
        X_train, X_test, y_train, y_test = train_test_split(df_train, df_test)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=10)

        print('Evaluating model...')
        forecasting = evaluate_model(model, X_test)

        original_testing_dataset['forecasted'] = forecasting

        print('Saving Forecasted Data...\n')
        original_testing_dataset.to_csv('data/forecasted_data.csv')

        plot_diagnostics(y_test, forecasting)

        print('\nSaving model...\n    MODEL: {}'.format(modelpath))
        save_model(model, modelpath)
        
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the cleaned training data '\
              'as the first argument, the filpath of the cleaned testing data as' \
              'the second argument and the filepath of the pickle file to '\
              'save the model to as the third argument. \n\nExample: python '\
              'train_forecaster.py training_data.csv testing_data.csv forcaster.pkl')

if __name__ == '__main__':
    main()