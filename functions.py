import pandas as pd


def arruma_dados(df):
    df.ds = pd.to_datetime(df.ds).dt.date
    df = df.rename(columns={'ds':'data'})
    df = df.sort_values(by=['loja', 'data'])
    df.data = pd.to_datetime(df.data)
    df['imputado'] = 0
    return df


def arruma_por_loja(df):
    dfs = []

    for i in df.loja.unique():
        dfl = df.loc[df.loja==i]
        dfl = preenche_datas_faltantes(dfl)
        dfl = preenche_dados_imputados(dfl)
        dfs.append(dfl)

    return pd.concat(dfs)


def preenche_datas_faltantes(df):
    semanas = pd.date_range(df.iloc[0, 3], df.iloc[-1, 3], freq='W')
    for semana in semanas:
        if df.loc[df.data==semana].empty:
            linha = pd.DataFrame({'item':[df.item.iloc[0]], 'loja':[df.loja.iloc[0]], 'tipo_loja':[df.tipo_loja.iloc[0]], 'data':[semana], 'venda':[0], 'imputado':[1]})
            df = pd.concat([df, linha], axis=0).sort_values(by='data').reset_index(drop=True)
    return df


def preenche_dados_imputados(df):
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


def limpa_dataframe(df):
    df = arruma_dados(df)
    df = arruma_por_loja(df)
    return df
