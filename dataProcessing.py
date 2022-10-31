import pandas as pd
from functions import *

class DataProcessor:

    def __init__(self):
        pass

    def read_data(self, csv_file):
        return pd.read_csv(csv_file)

    def clean(self, df, test=False):
        return clean_data(df, test)

    def features(self, df):
        return feature_engineer(df)