import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict

def fit_round(rnd: int)->Dict:
    return {"rnd": rnd}

def get_eval_fn(model: LogisticRegression):
    _, (X_test, y_test) = utils.load_mnist()

    def evaluate(parameters: fl.common.Weights):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}
    
    return evaluate

model = LogisticRegression()
utils.set_initial_params(model)
strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,
    eval_fn=get_eval_fn(model)
    on_fit_config_fn=fit_round,
)

fl.server.start_server(
    "0.0.0.0:8080",
    strategy=strategy,
    config={"num_rounds": 5}
)