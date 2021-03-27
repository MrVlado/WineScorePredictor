from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from time import time
import util as u
import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pickle

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def batch_grid(kernel, wine):
    if (kernel == 'rbf'):
        C_range = np.logspace(0, 4, 5)
        gamma_range = np.logspace(-2, 2, 5)
        param_grid = [
            {'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}]
    elif (kernel == 'poly'):
        C_range = np.logspace(-1, 3, 5)
        gamma_range = np.logspace(-2, 2, 5)
        param_grid = [
            {'C': C_range, 'gamma': gamma_range, 'kernel': ['poly'], 'degree': [ 4, 5,6], 'coef0': [0, 1, 2, 3, 4]}]
    elif (kernel == 'sigmoid'):
        C_range = np.logspace(3, 7, 5)
        gamma_range = np.logspace(-3, 1, 5)
        param_grid = [
            {'C': C_range, 'gamma': gamma_range, 'kernel': ['sigmoid'], 'coef0': [ -1, 0, 1, 2, 3]}]
    df = u.get_wine(wine)
    X_tr_norm, Y_tr, X_te_norm, Y_te = prepare_data(df)

    grid_search = model_selection.GridSearchCV(svm.SVR(epsilon=0.2, verbose=False, max_iter=100000),
                                               param_grid, n_jobs=2, verbose=2, scoring='neg_mean_absolute_error',return_train_score= True, cv=5)

    t0 = time()

    grid_search.fit(X_tr_norm, Y_tr)

    print("done in %0.3fs" % (time() - t0))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)

    u.savegrid(grid_search,kernel,wine)

    return grid_search



def prepare_data(df) :
    df = u.shuffle(df, 999)
    X, Y = u.xy(df)

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X)
    df_tr, df_te = u.split(df, 0.75)

    X_tr, Y_tr = u.xy(df_tr)
    X_te, Y_te = u.xy(df_te)

    X_te_norm = scaler.transform(X_te)
    X_tr_norm = scaler.transform(X_tr)
    return X_tr_norm, Y_tr, X_te_norm, Y_te