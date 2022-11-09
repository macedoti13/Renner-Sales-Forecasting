import pandas as pd

def fill_missing_weeks(df):
    """Adds a new row with the correct week, for every week that is missing in the data"""
    df['imputado'] = 0

    first_date = df.ds.min()
    last_date = df.ds.max()

    time = pd.date_range(first_date, last_date, freq='W')

    for week in time:
        if df.loc[df.ds == week].empty:
            if week < df.loc[(df.train==0)].ds.min():
                row = pd.DataFrame({'loja':[df.iloc[0,0]], 'tipo_loja':[df.iloc[0,1]], 'ds':[week], 'venda':[0], 'imputado':[1], 'train':[1]})
                df = pd.concat([df, row], axis=0)

    return df

def get_store_statistics(df):
    """Calculates the sales statistics and creates a new column for them"""
    media = df.loc[(df.imputado==0)&(df.train==1), 'venda'].mean()
    std = df.loc[(df.imputado==0)&(df.train==1), 'venda'].std()
    var = df.loc[(df.imputado==0)&(df.train==1), 'venda'].var()
    mediana = df.loc[(df.imputado==0)&(df.train==1), 'venda'].median()

    df['media_loja'] = round(media,2)
    df['var_loja'] = round(var,2)
    df['mediana_loja'] = round(mediana,2)
    df['std_loja'] = round(std, 2)

    return df


def get_time_features(df):
    df['mes'] = df.ds.apply(lambda x: x.month)
    df['weekofyear'] = df.ds.apply(lambda x: x.weekofyear)
    df['year'] = df.ds.apply(lambda x: x.year)
    return df


def fill_imputed_values(df):

    for i, j in df.loc[df.imputado==1].iterrows():
        ano = j.year
        sem = j.weekofyear
        media = df.media_loja.iloc[0]
        valor_ano_anterior = df.loc[(df.year==ano-1)&(df.weekofyear==sem), 'venda'].values

        if len(valor_ano_anterior) == 0:
            df.loc[(df.year==ano)&(df.weekofyear==sem), 'venda'] = media
        else:
            df.loc[(df.year==ano)&(df.weekofyear==sem), 'venda'] = valor_ano_anterior[0]

    return df

def get_top_week(df):
    """Gets the top week of the year (the one with the most sales) and the distance (in weeks) for it"""

    top_week = df.groupby(['weekofyear'])['venda'].max().sort_values(ascending=False).head(1).keys()[0]

    df['topweek'] = df['weekofyear'].apply(lambda x: 1 if x == top_week else 0)
    df['distance_topweek'] = df['weekofyear'].apply(lambda x: top_week - x)

    return df

def get_last_week_sales(df):
    df = df.sort_values(by='ds')
    df['lastweeksales'] = df['venda'].shift(1)
    df['lastweeksales'] = df['lastweeksales'].fillna(df.media_loja.iloc[0])
    df.loc[df.train==0, 'lastweeksales'] = df.mediana_loja.iloc[0]
    return df


def get_last_year_sales(df):
    df = df.sort_values(by='ds')
    df['lastyearsales'] = df['venda'].shift(52)
    df['lastyearsales'] = df['lastyearsales'].fillna(df.media_loja.iloc[0])
    return df

def get_last_year_month_mean(df):
    for i, j in df.iterrows():
        sem = j.weekofyear
        mes = j.mes
        ano = j.year
        mes_ano_passado = df.loc[(df.mes==mes)&(df.year==ano-1)].venda
        if mes_ano_passado.empty:
            df.loc[(df.year==ano)&(df.weekofyear==sem), 'lastyearmonthmean'] = df.media_loja.iloc[0]
        else:
            media_mes_ano_passado = mes_ano_passado.mean()
            df.loc[(df.year==ano)&(df.weekofyear==sem), 'lastyearmonthmean'] = media_mes_ano_passado

    return df

def get_features_from_storename(df):
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

    df['country'] = df['loja'].apply(get_country)
    df['state'] = df['loja'].apply(get_state)
    df['index'] = df['loja'].apply(get_index).astype('int64')

    return df

def get_holidays(df):
    holiday_weeks = [1, 15, 16, 17, 19, 36, 41, 44, 46, 51, 52]
    df['holiday'] = df['weekofyear'].apply(lambda x: 1 if x in holiday_weeks else 0)
    return df

def clean_train_test(df):
    train = df.loc[df.train==1]
    test = df.loc[df.train==0]
    train = train.drop('train', axis=1)
    test = test.drop('train', axis=1)
    return train, test

def process_data(df):
    dfs = []

    for loja in df.loja.unique():
        df_loja = df.loc[df.loja==loja]
        df_loja = fill_missing_weeks(df_loja)
        df_loja = get_store_statistics(df_loja)
        df_loja = get_time_features(df_loja)
        df_loja = fill_imputed_values(df_loja)
        df_loja = get_features_from_storename(df_loja)
        df_loja = get_top_week(df_loja)
        df_loja = get_last_year_sales(df_loja)
        df_loja = get_last_year_month_mean(df_loja)
        dfs.append(df_loja)

    df = pd.concat(dfs)
    df = get_holidays(df)
    df = df.sort_values(by=['loja', 'ds'])
    df.index = pd.to_datetime(df.ds)
    df = pd.get_dummies(df, columns=['tipo_loja', 'country', 'state'], prefix=['t','c','s'], drop_first=True)
    df = df.drop(columns=['imputado', 'ds'], axis=1)

    train, test = clean_train_test(df)

    return train, test