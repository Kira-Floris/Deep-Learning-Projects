import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

(X_train, y_train), (X_test, y_test) = utils.load_mnist()

partition_id = np.random.choice(10)
(X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

model = LogisticRegression(
    penalty="l2",
    max_iter=1,
    warm_start=True
)

utils.set_initial_params(model)

class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return utils.get_model_parameters(model)
    
    def fit(self, parameters, config):
        utils.set_model_params(model, parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            print(f'Training finished for round {config["rnd"]}')
        return utils.get_model_parameters(model), len(X_train), {}
    
    def evaluate(self, parameters, config):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}
    
fl.client.start_client(
    server_address="localhost:8080", 
    client=MnistClient().to_client()
)