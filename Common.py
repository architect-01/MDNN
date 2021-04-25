import numpy as np

def one_hot(y, n_classes):
    """ Convert the target(y) to it's one hot representation """
    
    y_one_hot = np.eye(n_classes)[y].T.reshape((n_classes, -1))
    
    return y_one_hot

def callback (dic):
    """ Callback passed to the model - used to print the information about the training progress """

    train_acc = np.sum(np.argmax(dic['y'], axis = 0) == np.argmax(dic['y_pred'], axis = 0))/ dic['y'].shape[-1]
    
    print(f'Epoch: {1+dic["epoch"]},\ttrain_cost: { dic["train_cost"]},\ttrain_acc = {train_acc}')


def evaluation_metric (dic):
    """ Callback passed to the model - used to print the information about the training progress """

    acc = np.sum(np.argmax(dic['y'], axis = 0) == np.argmax(dic['y_pred'], axis = 0))/ dic['y'].shape[-1]
    
    print(f'Test accuracy: {acc}')