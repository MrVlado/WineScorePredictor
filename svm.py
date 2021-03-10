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

C_range = np.logspace(-5, 9, 15)
gamma_range = np.logspace(-9, 5, 15)


# gamma_range = np.logspace(-7, 2, 10 )

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def score(kernel, wine):
    file_name = "grid_" + wine + "_" + kernel + ".pkl"
    with open(file_name, 'rb') as f:  # Python 3: open(..., 'wb')
        grid_search = pickle.load(f)
    params = grid_search.best_params_
    model = grid_search.best_estimator_
    # print(type(params['gamma']))
    df = u.get_wine("red")
    df = u.shuffle(df, 999)
    X, Y = u.xy(df)

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X)
    df_tr, df_te = u.split(df, 0.75)

    X_tr, Y_tr = u.xy(df_tr)
    X_te, Y_te = u.xy(df_te)

    X_te_norm = scaler.transform(X_te)

    Y_pred = model.predict(X_te_norm)

    return sklearn.metrics.mean_absolute_error(Y_te, Y_pred), model


def batch_grid(kernel, wine):
    if (kernel == 'rbf'):
        param_grid = [
            {'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}]
    elif (kernel == 'poly'):
        param_grid = [
            {'C': C_range, 'gamma': gamma_range, 'kernel': ['poly'], 'degree': [2, 3, 4, 5, 6], 'coef0': [0, 1, 2, 3, 4, 5]}]
    elif (kernel == 'sigmoid'):
        param_grid = [
            {'C': C_range, 'gamma': gamma_range, 'kernel': ['sigmoid'], 'coef0': [-2, -1, 0, 1, 2]}]
    df = u.get_wine(wine)
    df = u.shuffle(df, 999)
    df_train, df_test = u.split(df, 0.75)

    X, Y = u.xy(df)

    X_train, Y_train = u.xy(df_train)
    X_test, Y_test = u.xy(df_test)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_train)

    scaler = preprocessing.MinMaxScaler()

    scaler.fit(X)
    X_train_norm = scaler.transform(X_train)

    grid_search = model_selection.GridSearchCV(svm.SVR(epsilon=0.35, cache_size=3000, verbose=False, max_iter=100000),
                                               param_grid, n_jobs=-1, verbose=5, scoring='neg_mean_absolute_error')

    t0 = time()

    grid_search.fit(X_train_norm, Y_train)
    file_name = "grid_" + wine + "_" + kernel + ".pkl"
    with open(file_name, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(grid_search, f)

    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    return grid_search


def main():
    grid = batch_grid('poly', 'white')

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=-2, midpoint=-1))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

main()