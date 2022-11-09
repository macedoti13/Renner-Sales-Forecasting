from functions.etl import process_data
import warnings
import pandas as pd
import sys

warnings.filterwarnings('ignore')


def extract_data(train_data_filepath, test_data_filepath):
    """_summary_

    Args:
        train_data_filepath (_type_): _description_
        test_data_filepath (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df_train = pd.read_csv(train_data_filepath)
    df_test = pd.read_csv(test_data_filepath)

    return df_train, df_test


def transform_data(df_train, df_test):
    """_summary_

    Args:
        df_train (_type_): _description_
        df_test (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df_train['train'] = 1
    df_test['train'] = 0

    df = pd.concat([df_train, df_test], axis=0)

    df.ds = pd.to_datetime(df.ds).dt.date
    df.ds = pd.to_datetime(df.ds)

    df = df.drop('item', axis=1)

    df_train, df_test = process_data(df)

    return df_train, df_test


def load_data(df_train, df_test):
    """_summary_

    Args:
        df_train (_type_): _description_
        df_test (_type_): _description_
    """    
    df_train.to_csv('clean_data/train.csv')
    df_test.to_csv('clean_data/test.csv')


def main():
    if len(sys.argv) == 3:

        train_data_filepath, test_data_filepath = sys.argv[1:]

        print('Loading data...\n    TRAINING: {}\n    TESTING: {}'.format(train_data_filepath, test_data_filepath))
        df_train, df_test = extract_data(train_data_filepath, test_data_filepath)

        print('Cleaning data...')
        df_train, df_test = transform_data(df_train, df_test)

        print('Saving data...\n    FOLDER: clean_data')
        load_data(df_train, df_test)

        print('Cleaned data in folder!')
    
    else:
        print('Please provide the filepaths of the training and testing '\
              'datasets as the first and second argument respectively. \n\nExample: python3 \
              process_data.py data/training_data.csv data/testing_data.csv')


if __name__ == "__main__":
    main()