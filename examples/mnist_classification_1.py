"""
    Example of using the MDNN library on the the MNIST dataset


    MNIST dataset contains images of handwritten digits(all vs 1) of 28 x 28 x 1 size

"""
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pandas as pd

from Common import *
from Dense import *
from Activation import *
from Model import *

#load the training set into RAM
train_dataset = pd.read_csv('../dataset/mnist/mnist_train.csv')

n_classes = 1 ; image_width, image_height = 28, 28

X_train, y_train = train_dataset.iloc[:, 1:].T, np.array(train_dataset.iloc[:, :1] == 1, dtype=int).T

n_features, m_train = X_train.shape

print(X_train.shape, y_train.shape)


#hyperparams
n_epochs = 3000

#model definition
model = Model('categorical_cross_entropy')

model.add_layers([Dense(n_features, 51),
                  Activation('relu'),
                  Dense(51, n_classes),
                  Activation('sigmoid')])

model.fit(X_train, y_train, n_epochs, callback)

model.evaluate(X_test, y_test, evaluation_metric)