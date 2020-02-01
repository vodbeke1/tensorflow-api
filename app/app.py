import os
import pandas as pd
import numpy as np
import json
from flask import Flask
from flask import request
from flask import make_response
from env import Config


import keras
from keras.models import Model, Sequential

import tensorflow as tf

def load_params():
    if os.path.isfile("config/config.json") == False:
        return None
    with open("config/config.json", "r") as fp:
        params = json.load(fp)
    return params


def load_data():
    n_classes = 10
    train = pd.read_csv("data/mnist_train.csv")
    test = pd.read_csv("data/mnist_test.csv")

    x_train = train.iloc[:,1:].values/255
    x_test = test.iloc[:,1:].values/255

    y_train = keras.utils.to_categorical(train.iloc[:,0].values, n_classes)
    y_test = keras.utils.to_categorical(test.iloc[:,0].values, n_classes)

    return x_train, y_train, x_test, y_test


class ModelParams:
    def __init__(self, name, n_hidden, n_classes, n_input=None, **kwargs):
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
        self.verbose = kwargs.get("verbose", 1)


def build_model(params):

    layers = []

    for g in params.n_hidden:
        if g["type"] == "dense":
            layers.append(tf.keras.layers.Dense(g["neuron_count"], g["activation_type"]))
        elif g["type"] == "dropout":
            layers.append(tf.keras.layers.Dropout(g["dropout_rate"]))
        elif g["type"] == "flatten":
            pass
        elif g["type"] == "convolution":
            pass

    model = tf.keras.models.Sequential(layers)
    model.compile(loss=params.loss,
                  optimizer=params.optimizer,
                  metrics=params.metrics)

    model.save("model/{}.h5".format(params.name))
    return model





application = Flask(__name__)
application.config.from_object(Config)


@application.route("/score-object", methods=["POST"])
def score_object():
    pass


@application.route("/configure-model", methods=["POST"])
def configure_model():
    config = request.json
    custom = config.get("custom_config") 

    if type(custom) == str:
        with open("config/config-custom-{}.json".format(custom), "w") as fp:
            json.dump(config, fp)
        return make_response("Custom parameters set successfully - reference:{}".format(custom), 200) 

    with open("config/config.json", "w") as fp:
        json.dump(config, fp)

    return make_response("Parameters set successfully", 200)
    

@application.route("/train-model", methods=["GET"])
def train_model():

    x_train, y_train, x_test, y_test = load_data()
    params = load_params()

    if params == None:
        return make_response("No model parameters found", 201)

    params = ModelParams(name=params["name"], 
                        n_classes=params["n_classes"], 
                        n_hidden=params["layers"], 
                        training_epochs=params["training_epochs"],
                        loss=params["loss"],
                        optimizer=params["optimizer"],
                        metrics=params["metrics"])
    
    model = build_model(params)
    
    model.fit(x_train, y_train,
        batch_size=params.batch_size,
        epochs=params.training_epochs,
        verbose=params.verbose,
        validation_data=(x_test,y_test))

    model.summary()
    print(model.evaluate(x_test, y_test))
    model.save("model/{}.h5".format(params.name))

    return make_response("Model trained successfully", 200)


if __name__ == "__main__":
    application.run()