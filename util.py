import pandas as pd
import math
import numpy as np
import pickle
import nn
import svm
import sklearn
from sklearn import decomposition
import plotly.express as px

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
    return  df.iloc[:i], df.iloc[i:]

def loadgrid(kernel, wine, nnpath=None):
    # grid_red_poly.pkl
    # gird_white_nn_layers_activation
    file_name = 'grid_' + wine + '_' + kernel + (nnpath != None and ('_' + nnpath) or "") + '.pkl'
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def savegrid(grid_search, kernel, wine, nnpath=None):
    file_name = 'grid_' + wine + '_' + kernel + (nnpath != None and ('_' + nnpath) or "") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(grid_search, f)

def score(kernel, wine,nnpath = None,T=1):
    grid_search = loadgrid(kernel,wine,nnpath)
    model = grid_search.best_estimator_
    if (kernel == 'nn') :
        X_train, Y_train, X_test, Y_test = nn.prepare_data(get_wine(wine))
    else :
        X_train, Y_train, X_test, Y_test = svm.prepare_data(get_wine(wine))

    Y_pred = model.predict(X_test)

    return sklearn.metrics.mean_absolute_error(Y_test, Y_pred), model, np.sum(abs((Y_pred-Y_test))<T)/Y_pred.shape[0]


def rec(kernel, wine, nnpath=None):
    grid_search = loadgrid(kernel,wine,nnpath)
    model = grid_search.best_estimator_
    if (kernel == 'nn') :
        X_train, Y_train, X_test, Y_test = nn.prepare_data(get_wine(wine))
    else :
        X_train, Y_train, X_test, Y_test = svm.prepare_data(get_wine(wine))

    Y_pred = model.predict(X_test)

    def acc(i):
        return np.sum(abs((Y_pred-Y_test))<i)/Y_pred.shape[0]

    x = np.linspace(0, 2, 100)
    y = [ acc(i) for i in x ]

    #return px.line(x=x, y=y)
    return x, y
    
def pca2d(df, showloadings=False, norm= 'none'):
    if (norm == 'z_score') :
        df_norm = (df-df.mean())/df.std()

    elif (norm == 'min_max') :
        df_norm = (df-df.min())/(df.max()-df.min())
    else :
        df_norm = df
    pca = sklearn.decomposition.PCA(n_components=2)
    components = pca.fit_transform(df_norm.iloc[:, :-1])

    fig = px.scatter(components, x=0, y=1, color=df_norm.quality, labels={
        '0': str(pca.explained_variance_ratio_[0]),
        '1': str(pca.explained_variance_ratio_[1]),
        'color': 'quality'
    }, title=f'Total Explained Variance: {pca.explained_variance_ratio_.sum()*100:.2f}%')

    if showloadings:
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_) * 10
        for i, feature in enumerate(df_norm.columns[:-1]):
            fig.add_shape(
                type='line',
                x0=0, y0=0,
                x1=loadings[i, 0],
                y1=loadings[i, 1]
            )
            fig.add_annotation(
                x=loadings[i, 0],
                y=loadings[i, 1],
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
            )

    return fig.show()

def pcagrid(df, n_components=4,norm= 'none'):
    if (norm == 'z_score') :
        df_norm = (df-df.mean())/df.std()
    elif (norm == 'min_max') :
        df_norm = (df-df.min())/(df.max()-df.min())
    else :
        df_norm = df
        
    pca = sklearn.decomposition.PCA(n_components=n_components)
    components = pca.fit_transform(df.iloc[:, :-1])
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    labels['color'] = 'quality'
    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(len(pca.explained_variance_ratio_)),
        color=df["quality"],
        title=f'Total Explained Variance: {pca.explained_variance_ratio_.sum()*100:.2f}%'
    )
    fig.update_traces(diagonal_visible=False)
    return fig.show()
