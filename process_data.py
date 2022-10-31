from functions import *
from dataProcessing import DataProcessor
import sys

def extract_data(train_dataset_filepath, test_dataset_filepath, data_processor):
    df_train = data_processor.read_data(train_dataset_filepath)
    df_test = data_processor.read_data(test_dataset_filepath)
    return df_train, df_test

def transform_data(train_dataset, test_dataset, data_processor):
    df_train = data_processor.clean(train_dataset)
    df_test = data_processor.clean(test_dataset, test=True)

    df_train = data_processor.features(df_train)

    store_avgs = df_train.groupby('loja')['venda'].mean()
    store_stds = df_train.groupby('loja')['venda'].std()    

    df_test = data_processor.features(df_test, test=True, store_avgs=store_avgs, store_stds=store_stds)

    return df_train, df_test

def load_data(train_dataset, test_dataset):
    train_dataset.to_csv('cleaned_data/cleaned_training_data.csv')
    test_dataset.to_csv('cleaned_data/cleaned_testing_data.csv')

def main():

    dt = DataProcessor()

    if len(sys.argv) == 3:

        train_dataset_filepath, test_dataset_filepath = sys.argv[1:]

        print('Loading data...\n    TRAINING: {}\n    TESTING: {}'.format(train_dataset_filepath, test_dataset_filepath))
        df_train, df_test = extract_data(train_dataset_filepath, test_dataset_filepath, dt)

        print('Cleaning data...')
        df_train, df_test = transform_data(df_train, df_test, dt)

        print('Saving data...\n    FOLDER: cleaned_data')
        load_data(df_train, df_test)

        print('Cleaned data in folder!')
    
    else:
        print('Please provide the filepaths of the training and testing '\
              'datasets as the first and second argument respectively. \n\nExample: python process_data.py '\
              'training_data.csv testing_data.csv')

if __name__ == '__main__':
    main()