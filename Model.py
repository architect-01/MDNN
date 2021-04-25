import numpy as np

from Loss import *

class Model:
    """ Wrapper to encapsulate the forward prop and backward prop of the model """
    def __init__(self, loss_func):
        """Constructor
        
        Parameters
        ----------
        func_name : str
            Name of the loss function.
        
        Returns
        -------
        None
        """
        self.layers = []
        self.loss_func = Loss(loss_func)
    
    def add_layers(self, layers):
        """Stacks the passed layers to the existing ones
        
        Parameters
        ----------
        layers : list
            List of layers
        
        Returns
        -------
        None
        """
        self.layers += layers

            
    def fit(self, X, y, n_epochs, callback = None):
        """Trains the model
        
        Parameters
        ----------
        X : np.array
            Training data (features x examples)
        y : np.array
            Training target (labels x examples)
        callback: function that expects a dictionary
            Used to print the training progress information
        
        Returns
        -------
        None
        """
        m = X.shape[-1]
        
        for epoch in range(n_epochs):
    
            y_pred = self.predict(X)
        
            # calculate cost
            C = self.loss_func.calc_loss(y, y_pred)
            
            # backward pass
            grad = self.loss_func.calc_grad(y, y_pred)

            for layer in reversed(self.layers):
                grad = layer.backward_prop(grad)
                 
            #print progress information
            if callback:
                callback({'y': y, 'y_pred' : y_pred, 'epoch' : epoch, 'train_cost': C})
            
    def predict(self, X):
        """Does forward prop - a.k.a predicts the target
        
        Parameters
        ----------
        X : np.array
            Data (features x examples)

        Returns
        -------
        np.array
            Predictions
        """
        Z = X
        for layer in self.layers:
            Z = layer.forward_prop(Z)
            
        return Z
    
    def evaluate(self, X, y, callback):
        """Evaluates the model on (X, y) dataset
        
        Parameters
        ----------
        X : np.array
            Data (features x examples)
        y : np.array
            Target (labels x examples)
        callback: function
            Evaluation metric
        
        Returns
        -------
        None
        """
        
        y_pred = self.predict(X)
        
        callback({'y': y, 'y_pred': y_pred})