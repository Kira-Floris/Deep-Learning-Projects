from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import openml

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def get_model_parameters(
        model:LogisticRegression
    )->Tuple:
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef,)
    return params

def set_model_params(
        model:LogisticRegression, 
        params: LogRegParams
    )->LogisticRegression:
    model.coef_ = params
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(
        model:LogisticRegression
    ):
    n_classes = 10
    n_features = 784
    model.classes_ = np.array([i for i in range(10)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def load_mnist()->Dataset:
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    X = Xy[:, :-1]
    y = Xy[:, -1]
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test)

def partition(X:np.ndarray, y:np.ndarray, num_partitions:int)->XYList:
    return list(
        zip(np.array_split(X, num_partitions),
            np.array_split(y, num_partitions))
    )