"""
    Example of using the MDNN library on the the MNIST dataset - CNN model


    MNIST dataset contains images of handwritten digits(0-9) of 28 x 28 x 1 size

"""
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pandas as pd

from Common import *
from Dense import *
from Convolution import *
from Padding import *
from Flatten import *
from Activation import *
from Model import *

#load the training set into RAM
train_dataset = pd.read_csv('../dataset/mnist/mnist_train.csv')

n_classes = 10 ; image_width, image_height = 28, 28
m_train = 500

#reshape the set from 784 to 28x28
X_train = np.reshape(train_dataset.iloc[:m_train, 1:].to_numpy(), (-1, image_height, image_width, 1)).transpose(1, 2, 3, 0)
y_train = one_hot(np.array(train_dataset.iloc[:m_train, :1]).T, n_classes)

_, _2, _3, m_train = X_train.shape

print(X_train.shape, y_train.shape)

#hyperparams
n_epochs = 300

#model definition
model = Model('categorical_cross_entropy')

model.add_layers([Padding(2),
                  Convolution([32, 32, 1], [5, 5, 1], 31),
                  Activation('relu'),
                  Padding(2),
                  Convolution([14, 14, 31], [3, 3, 31], 22, step_size = 2),
                  Activation('relu'),
                  Flatten(),
                  Dense(792, 51),
                  Activation('relu'),
                  Dense(51, n_classes),
                  Activation('softmax')])

model.fit(X_train, y_train, n_epochs, callback)

model.evaluate(X_test, y_test, evaluation_metric)