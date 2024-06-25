import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def get_shakespeare():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

def prepare_dataset(num_partitions:int, batch_size=32, val_ratio=0.1):
    (x_train, y_train), (x_test, y_test) = get_shakespeare()

    num_images = len(x_train) // num_partitions
    train_partitions = []
    
    for i in range(num_partitions):
        x_part = x_train[i*num_images:(i+1)*num_images]
        y_part






