"""
    Example of using the MDNN library on the Iris dataset


    Iris dataset contains three species of iris flowers for the total number of 150 examples.
    
    Each example contains four measurements of the species.
    
    Note:
        This dataset is too small for Neural Network to be used and some other ML algorithm would give better results

"""
#needed as to import from the parent directory
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pandas as pd
import numpy as np

from Common import *

from Dense import *
from Activation import *

from Model import *

#setting the seed for reproducability of results
np.random.seed(1)

#load and shuffle the dataset
dataset = pd.read_csv('../dataset/iris.csv').sample(frac=1).reset_index(drop=True)

n_classes = 3; m = 149
#separate data and the target

#train, test, val split percentages
m_train = int(0.7*m) ; m_test = m_train + int(0.1*m) ; m_val = -1

X_train, y_train = dataset.iloc[:m_train, :-1].T.to_numpy(), one_hot(dataset.iloc[:m_train, -1:], n_classes)
X_test, y_test = dataset.iloc[m_train:m_test, :-1].T.to_numpy(), one_hot(dataset.iloc[m_train:m_test, -1:], n_classes)

#number of features and number of examples(size of the dataset)
n_features, m = X_train.shape ;

#hyperparams
n_epochs = 300000

#model definition
model = Model('categorical_cross_entropy')

model.add_layers([Dense(n_features, 51),
                  Activation('relu'),
                  Dense(51, n_classes),
                  Activation('softmax')])

model.fit(X_train, y_train, n_epochs, callback)

print(X_train.shape)

model.evaluate(X_test, y_test, evaluation_metric)