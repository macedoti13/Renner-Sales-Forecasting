import pandas as pd
import sys
from functions.ml import *

def load_data(train, test):
    df_train = pd.read_csv(train, index_col='ds')
    df_test = pd.read_csv(test, index_col='ds')
    return df_train, df_test

def main():
    if len(sys.argv) == 5:

        train_data, test_data, original_testing_dataset, modelpath= sys.argv[1:]
        print('Loading data...\n')

        df_train, df_test = load_data(train_data, test_data)
        original_testing_dataset = pd.read_csv(original_testing_dataset)

        print('Building training and testing datasets...\n')
        X_train, X_test, y_train, y_test = train_test_split(df_train, df_test)

        print('Building model...')
        model = build_model()

        print('Training model...')
        train_model(model, X_train, X_test, y_train, y_test)

        print('Evaluating model...')
        forecast = evaluate_model(model, X_test, y_test)
        plot_diagnostics(y_test, forecast)

        original_testing_dataset = original_testing_dataset.sort_values(by=['loja', 'ds'])
        original_testing_dataset['ypred'] = forecast
        original_testing_dataset['ypred'] = np.abs(original_testing_dataset['ypred'])
        original_testing_dataset['error'] = original_testing_dataset['venda'] - original_testing_dataset['ypred']
        original_testing_dataset['sqderror'] = original_testing_dataset['error']**2
        original_testing_dataset['abserror'] = original_testing_dataset['error'].apply(lambda x: np.abs(x))

        print('Saving Forecasted Data...\n')
        original_testing_dataset.to_csv('data/forecasted_data.csv')

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