import pandas as pd
import numpy as np
from flask import Flask

import keras
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam

import tensorflow as tf



def load_data():
    train = pd.read_csv("data/mnist_train.csv")
    test = pd.read_csv("data/mnist_test.csv")

    train_images = train.iloc[:,:-1].values
    test_images = test.iloc[:,:-1].values

    train_label = train.iloc[:,-1].values
    test_label = test.iloc[:,-1].values

    return train_images, test_images, train_label, test_label

class ModelParams:
    def __init__(self, name, n_input, n_hidden, n_classes, **kwargs):
        # Name
        self.name = name
        
        # Network Parameters
        self.n_input = n_input # MNIST data input (img shape: 28*28 flattened to be 784)
        self.n_hidden = n_hidden # Number  of neurons
        self.n_classes = n_classes # Number of classes of prediction

        # Compile Parameters
        self.loss = kwargs.get("loss", "categorical_crossentropy") 
        self.optimizer = kwargs.get("optimizer", "SGD")
        self.metrics = kwargs.get("metrics", ["accuracy"])

        # Training Parameters for basic MNIST
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.training_epochs = kwargs.get("training_epochs", 2)
        self.batch_size = kwargs.get("batch_size", 100)

def build_model(params):

    layers = []

    for g in params.n_hidden:
        if g["type"] == "dense":
            layers.append(tf.keras.layers.Dense(g["neuron_count"], g["activation_type"]))
        elif g["type"] == "dropout":
            layers.append(tf.keras.layers.Dropout(g["dropout_rate"]))
        elif g["type"] == "flatten":
            pass
    model = tf.keras.models.Sequential(layers)
    model.compile(loss=params.loss,
              optimizer=params.optimizer,
              metrics=params.metrics)
    model.save("model/{}.h5".format(params.name))
    return model

application = Flask(__name__)

@application.route("/score-object", methods=["POST"])
def score_object():
    pass

@application.route("/configure-model", methods=["POST"])
def configure_model():
    pass 

@application.route("/train-model", methods=["GET"])
def train_model():
    pass


if __name__ == "__main__":
    application.run()



