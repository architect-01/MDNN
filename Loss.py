import numpy as np


class Loss:
    """ Define loss function """
    
    def __init__(self, func_name = 'categorical_cross_entropy'):
        """Sets the loss function
        
        Parameters
        ----------
        func_name : str
            Name of the loss function.
        
        Returns
        -------
        None
        """
        
        if func_name == 'categorical_cross_entropy':
            self.calc_loss = self._categorical_cross_entropy;  
            self.calc_grad = self._grad_categorical_cross_entropy
      
    def _categorical_cross_entropy(self, y, y_pred):
        """Calculate the loss of the batch
        
        Parameters
        ----------
        y : np.array
            True target
        y_pred : np.array
            Predicted target
            
        Returns
        -------
        np.array
            Loss on the batch.
        """
    
        return - np.sum(y * np.log(y_pred + 1e-8))
        
    def _grad_categorical_cross_entropy(self, y, y_pred):
        """Calculates gradient used to update the parameters of the model
        
        Parameters
        ----------
        y : np.array
            True target
        y_pred : np.array
            Predicted target
            
        Returns
        -------
        np.array
            Gradient to be passed to the next layers(looking from the end to the start of the model)
        """
    
        return  - y / ( 1e-8 + y_pred );
