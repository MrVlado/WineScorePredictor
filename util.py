import pandas as pd
import math
import numpy as np
import pickle


def get_wine(kind):
    if kind not in ['red', 'white']:
        raise ValueError('there are only red and white wines..')
    return pd.read_csv('DATASET_WINE/winequality-' + kind + '.csv', sep=';')


def xy(df):
    return np.ascontiguousarray(df.iloc[:, :-1].to_numpy()), np.ascontiguousarray(df['quality'].to_numpy())


def shuffle(df, seed):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def split(df, trainshare):
    i = math.floor(df.shape[0] * trainshare)
    return df.iloc[i:], df.iloc[:i]

def loadgrid(kernel, wine, nnpath=None):
    # grid_red_poly.pkl
    # gird_white_nn_layers_activation
    file_name = 'grid_' + wine + '_' + kernel + (nnpath != None and ('_' + nnpath) or "") + '.pkl'
    with open(file_name, 'rb') as f:
        grid_search = pickle.load(f)
    return gird_search

def savegrid(kernel, wine, nnpath=None):
    file_name = 'grid_' + wine + '_' + kernel + (nnpath != None and ('_' + nnpath) or "") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(grid_search, f)
