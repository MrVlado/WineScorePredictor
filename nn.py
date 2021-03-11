# There's no sense to use more than 2 hidden layers , every model can be rappresented by 2 hidden layers
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
import util as u
import numpy as np
import pickle
from time import time
from sklearn import preprocessing


def prepare_data(df):
    df = u.shuffle(df, 999)
    df_train, df_test = u.split(df, 0.75)

    X_train, Y_train = u.xy(df_train)
    X_test, Y_test = u.xy(df_test)

    X_train = preprocessing.maxabs_scale(X_train)
    X_test = preprocessing.maxabs_scale(X_test)

    ones = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((X_train, ones))

    ones = np.ones((X_test.shape[0], 1))
    X_test = np.hstack((X_test, ones))

    return X_train, Y_train, X_test, Y_test


def grid_search_layers(grid_search, wine):
    min_1 = 10
    min_2 = 10

    max_1 = 20
    max_2 = 20

    tuples = []
    for l1 in np.linspace(min_1, max_1, 10, dtype=int):
        for l2 in np.linspace(min_2, max_2, 10, dtype=int):
            if l2 == 0:
                tuples.append((l1,))
            else:
                tuples.append((l1, l2))

    #layers_grid = [{'hidden_layer_sizes': tuples, 'activation': ['logistic', 'tanh', 'relu']}]
    layers_grid = [{'hidden_layer_sizes': tuples, 'activation': ['relu']}]

    mlp = MLPRegressor(max_iter=10000, solver='adam', tol=1e-4, random_state=999)

    if (grid_search == None):
        grid_search = model_selection.GridSearchCV(mlp, layers_grid, return_train_score=True, cv=5,
                                                   scoring='neg_mean_absolute_error', n_jobs=-1, verbose=3)
    else:
        grid_search.set_params(**layers_grid)
    X_train, Y_train, X_test, Y_test = prepare_data(u.get_wine(wine))

    t0 = time()

    grid_search.fit(X_train, Y_train)
    u.savegrid(grid_search, 'nn', wine, 'activation_layers')

    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    return grid_search


def grid_search_alpha(prev_grid, wine):
    mlp = MLPRegressor(**prev_grid.best_estimator_.get_params())
    X_train, Y_train, X_test, Y_test = prepare_data(u.get_wine(wine))

    alpha_grid = np.logspace(-5, -3, 100)

    params_grid = {'alpha': alpha_grid}

    grid_search = model_selection.GridSearchCV(mlp, params_grid, return_train_score=True, cv=5,
                                               scoring='neg_mean_absolute_error', n_jobs=-1, verbose=5)
    t0 = time()

    grid_search.fit(X_train, Y_train)

    u.savegrid(grid_search, "nn", wine, "alpha")
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    return grid_search


def grid_search_batch_size(prev_grid, wine):
    mlp = MLPRegressor(**prev_grid.best_estimator_.get_params())
    X_train, Y_train, X_test, Y_test = prepare_data(u.get_wine(wine))
    params_grid = {'batch_size': np.linspace(1, X_train.shape[0], 200, dtype=int)}
    grid_search = model_selection.GridSearchCV(mlp, params_grid, return_train_score=True, cv=5,
                                               scoring='neg_mean_absolute_error', n_jobs=-1, verbose=5)
    t0 = time()

    grid_search.fit(X_train, Y_train)

    u.savegrid(grid_search, "nn", wine, "batch_size")
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    return grid_search


def grid_search_lr(prev_grid, wine):
    prev_params = prev_grid.best_estimator_.get_params()
    prev_params['max_iter'] = 10000
    mlp = MLPRegressor(**prev_params)
    X_train, Y_train, X_test, Y_test = prepare_data(u.get_wine(wine))
    params_grid = {'learning_rate_init': np.logspace(-5, 0, 100)}
    grid_search = model_selection.GridSearchCV(mlp, params_grid, return_train_score=True, cv=5,
                                               scoring='neg_mean_absolute_error', n_jobs=-1,
                                               verbose=5)
    t0 = time()

    grid_search.fit(X_train, Y_train)

    u.savegrid(grid_search, "nn", wine, "lr")
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    return grid_search


def grid_search_beta(prev_grid, wine):
    prev_params = prev_grid.best_estimator_.get_params()
    prev_params['max_iter'] = 10000
    mlp = MLPRegressor(**prev_params)
    X_train, Y_train, X_test, Y_test = prepare_data(u.get_wine(wine))
    params_grid = {'beta_1': np.linspace(0.5, 0.9999999999, 20), 'beta_2': np.linspace(0.51, 0.99999999999, 20)}
    grid_search = model_selection.GridSearchCV(mlp, params_grid, return_train_score=True, cv=5,
                                               scoring='neg_mean_absolute_error', n_jobs=-1,
                                               verbose=5)
    t0 = time()

    grid_search.fit(X_train, Y_train)

    u.savegrid(grid_search, "nn", wine, "beta")
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    return grid_search


def grid_search_alpha_lr(params, wine):
    params_grid = {'alpha': np.logspace(-5, -3, 100), 'learning_rate_init': np.linspace(0.1, 1, 10)}
    mlp = MLPRegressor(**params)
    X_train, Y_train, X_test, Y_test = prepare_data(u.get_wine(wine))
    grid_search = model_selection.GridSearchCV(mlp, params_grid, return_train_score=True, cv=5,
                                               scoring='neg_mean_absolute_error', n_jobs=-1,
                                               verbose=1)
    t0 = time()

    grid_search.fit(X_train, Y_train)

    u.savegrid(grid_search, "nn", wine, "alpha_lr")
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    return grid_search
