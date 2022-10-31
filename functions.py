import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# data cleaning 

def _fill_missing_dates(df):
    weeks = pd.date_range(df.iloc[0, 3], df.iloc[-1, 3], freq='W')
    for week in weeks:
        if df.loc[df.data==week].empty:
            linha = pd.DataFrame({'item':[df.item.iloc[0]], 'loja':[df.loja.iloc[0]], 'tipo_loja':[df.tipo_loja.iloc[0]], 'data':[week], 'venda':[0], 'imputado':[1]})
            df = pd.concat([df, linha], axis=0).sort_values(by='data').reset_index(drop=True)
    return df

def _fill_imputed_data(df):
    for i in df.loc[(df.imputado==1)].index:
        m = 0
        j = 0
        if i > 53:
            if df.loc[i-53, 'loja']==df.loc[i, 'loja']:
                v1 = df.loc[i-53].venda
                m += v1
                j += 1
        if i > 105:
            if df.loc[i-105, 'loja']==df.loc[i, 'loja']:
                v2 = df.loc[i-105].venda
                m += v2
                j += 1
        if i > 157:
            if df.loc[i-157, 'loja']==df.loc[i, 'loja']:
                v3 = df.loc[i-157].venda
                m += v3
                j += 1

        if j > 0:
            m //= j
            
        df.loc[i, 'venda'] = m

    return df

def _imput_missing_data(df):
    dfs = []

    for i in df.loja.unique():
        dfl = df.loc[df.loja==i]
        dfl = _fill_missing_dates(dfl)
        dfl = _fill_imputed_data(dfl)
        dfs.append(dfl)

    return pd.concat(dfs)

def _fix_data(df):
    df.ds = pd.to_datetime(df.ds).dt.date
    df = df.rename(columns={'ds':'data'})
    df = df.sort_values(by=['loja', 'data'])
    df.data = pd.to_datetime(df.data)
    df['imputado'] = 0
    return df

def clean_data(df, test = False):
    df = _fix_data(df)
    if not test:
        df = _imput_missing_data(df)
    return df


# features engineering

def get_country(str):
    parts = str.split('_')[1:]
    country = parts[0]
    return country

def get_state(str):
    parts = str.split('_')[1:]
    state = parts[1]
    return state

def get_index(str):
    parts = str.split('_')[1:]
    index = parts[2]
    return index

def get_week(date):
    return date.weekofyear

def get_year(date):
    return date.year

def store_avg(store, stores_avg):
    return stores_avg[store]

def store_std(store, stores_var):
    return stores_var[store]

def feature_engineer(df):
    df = df.reset_index(drop=True)
    df = df.drop(columns=['item', 'imputado'])
    df = pd.get_dummies(df,columns=['tipo_loja'],drop_first=True, prefix='loja')
    df['country'] = df['loja'].apply(get_country)
    df['state'] = df['loja'].apply(get_state)
    df['index'] = df['loja'].apply(get_index)
    df = pd.get_dummies(df,columns=['country','state'],drop_first=True, prefix=['c','s'])
    df.data = pd.to_datetime(df.data)
    df['week_of_year'] = df.data.apply(get_week)
    df['year'] = df.data.apply(get_year)
    stores_avg = df.groupby('loja')['venda'].mean()
    df['store_avg'] = df['loja'].apply(lambda x: store_avg(x, stores_avg))
    stores_var = df.groupby('loja')['venda'].std()
    df['store_std'] = df['loja'].apply(lambda x: store_std(x, stores_var))
    holiday_weeks = [1, 15, 16, 17, 36, 41, 44, 46, 51, 52]
    df['holiday'] = df['week_of_year'].apply(lambda x: 1 if x in holiday_weeks else 0)

    return df

def feature_engineer_test(df, store_avgs, store_stds):
    df = df.reset_index(drop=True)
    df = df.drop(columns=['item', 'imputado'])
    df = pd.get_dummies(df,columns=['tipo_loja'],drop_first=True, prefix='loja')
    df['country'] = df['loja'].apply(get_country)
    df['state'] = df['loja'].apply(get_state)
    df['index'] = df['loja'].apply(get_index)
    df = pd.get_dummies(df,columns=['country','state'],drop_first=True, prefix=['c','s'])
    df.data = pd.to_datetime(df.data)
    df['week_of_year'] = df.data.apply(get_week)
    df['year'] = df.data.apply(get_year)
    df['store_avg'] = df['loja'].apply(lambda x: store_avgs[x])
    df['store_std'] = df['loja'].apply(lambda x: store_stds[x])
    holiday_weeks = [1, 15, 16, 17, 19, 36, 41, 44, 46, 51, 52]
    df['holiday'] = df['week_of_year'].apply(lambda x: 1 if x in holiday_weeks else 0)

    return df


# model evaluation 

def mse(y_real, y_pred):
    return mean_squared_error(y_real ,y_pred)

def rmse(y_real, y_pred):
    return np.sqrt(mse(y_real, y_pred))

def mae(y_real, y_pred):
    return mean_absolute_error(y_real, y_pred)

def plot_diagnostics(y_real, y_pred):
    print(f'MSE: {mse(y_real, y_pred)}')
    print(f'RMSE: {rmse(y_real, y_pred)}')
    print(f'MAE: {mae(y_real, y_pred)}')