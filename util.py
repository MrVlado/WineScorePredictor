import pandas as pd
import math

def get_wine(kind):
    if kind not in ['red', 'white']:
        raise ValueError('there are only red and white wines..')
    pd.read_csv('DATASET_WINE/winequality-'+kind+'.csv', sep=';')

def xy(df):
    return df.iloc[:,:-1].to_numpy().T, df['quality'].to_numpy().T

def shuffel(df):
    return df.sample(frac=1).reset_index(drop=True)

def split(df, trainshare):
    i = math.floor(df.shape[0] * trainshare)
    return df.iloc[i:], df.iloc[:i]
