import numpy as np

class Flatten:
    
    """layer that flatten it's input - it's output is used with Dense layer"""
    
    def __init__(self):
        """Constructor - no arguments are passed because none are needed
        
        Parameters
        ----------
        
        Returns
        -------
        None
        """
        
        pass
    
    
    def forward_prop(self, Z):
        """Computes forward propagation - Flattens 4D array into 2D array
        
        Parameters
        ----------
        Z : 4D np.array
            Output from the Convolutional layer.
        
        Returns
        -------
        2D np.array
            To be used as input by the Dense layer
        """
        
        self.shape = Z.shape #remember the shape - so that backward prop can reshape the gradien
        
        return Z.reshape(Z.shape[0]*Z.shape[1]*Z.shape[2], -1)
    
    def backward_prop(self, grad):
        """Computes backward propagation - reshapes gradient received from the higher layers
        
        Parameters
        ----------
        grad : 2D np.array
            gradient comming from the higher layers
        
        Returns
        -------
        4D np.array
            Reshaped gradient
        """
        
        return grad.reshape(self.shape)