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


def grid_hidden_layers(grid_search, wine):
    mlp = MLPRegressor(max_iter=1000, alpha=1e-4, solver='adam',
                       tol=1e-4, random_state=999,
                       learning_rate_init=.1)

    if (grid_search == None):
        grid_search = model_selection.GridSearchCV(mlp, layers_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
                                                   verbose=5)
    else:
        grid_search.set_params(**layers_grid)
    X_train, Y_train, X_test, Y_test = prepare_data(u.get_wine(wine))

    t0 = time()

    grid_search.fit(X_train, Y_train)
    file_name = "grid_" + wine + "_NN_layers_activation" + ".pkl"
    with open(file_name, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(grid_search, f)

    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    return grid_search


min_1 = 2
min_2 = 0

delta_1 = 1
delta_2 = 1

max_1 = 51
max_2 = 51

tuples = []
for l1 in range(min_1, max_1, delta_1):
    for l2 in range(min_2, max_2, delta_2):
        if l2 == 0:
            tuples.append((l1,))
        else:
            tuples.append((l1, l2))

layers_grid = [{'hidden_layer_sizes': tuples, 'activation': ['logistic', 'tanh', 'relu']}]

grid_hidden_layers(None, "red")
