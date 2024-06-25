from collections import OrderedDict
from typing import Dict, Tuple
import flwr as fl
import tensorflow as tf
from flwr.common import NDArrays, Scalar

from utils.model import NextWordModel, train, test

class Client(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model:tf.keras.Model = NextWordModel(vocab_size, embedding_dim, rnn_units)
        
    def get_parameters(self):
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        train(self.model, self.train_loader, config.optimizer, config.loss)
        return self.get_parameters({}), len(self.train_loader), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = test(self.model, self.val_loader)
        return loss, len(self.val_loader), {"accuracy": accuracy}
    

def generate_client_function(train_loaders, val_loaders, vocab_size, embedding_dim, rnn_units):
    def client_fn(cid:str):
        return Client(
            train_loader=train_loaders[int(cid)],
            val_loader=val_loaders[int(cid)],
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units
        )
    return client_fn