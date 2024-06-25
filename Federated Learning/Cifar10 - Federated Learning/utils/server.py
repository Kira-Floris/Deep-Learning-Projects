from collections import OrderedDict
from omegaconf import DictConfig
import tensorflow as tf
from utils.model import NextWordModel, test

def get_on_fit_config(config):
    def fit_config_fn(server_round):
        return {
            "lr":config.lr, "momentum":config.momentum, "local_epochs":config.local_epochs
        }
    return fit_config_fn

def get_evaluate_fn(vocab_size, test_loader, embedding_dim, rnn_units):
    def evaluate_fn(server_round, parameters, config):
        model = NextWordModel(vocab_size, embedding_dim, rnn_units)
        model.set_weights(parameters)
        loss, accuracy = test(model, test_loader)
        return loss, {"accuracy": accuracy}
    return evaluate_fn
