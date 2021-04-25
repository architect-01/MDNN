import numpy as np


class Padding:
    
    """Padding layer - used before Convolutional layer"""
    
    def __init__(self, padding):
        """Constructor
        
        Parameters
        ----------
        padding : int
            Number of pixels to be added height and widthwise.
            
        Returns
        -------
        None
        """
            
        self.hyperparameters = {'padding': padding}
    
        self.cache = {'O': None}
    
    def forward_prop(self, Z):
        """Computes forward propagation - zero padds it's input along two dimensions representing height and width
        
        Parameters
        ----------
        Z : 4D np.array
            Input to be padded
            
        Returns
        -------
        4D np.array
            Padded input.
        """
        padding = self.hyperparameters['padding']
        
        #NOTE: before and after padding over two axis is the same, TODO: add the possibility for them to be different
        self.cache['O'] = np.pad(Z, ((padding, padding),(padding, padding), (0, 0), (0, 0)),mode='constant')
    
        return self.cache['O']
    
    def backward_prop(self, grad):
        """Computes backward propagation - crops away the unnecessary gradients
        
        Parameters
        ----------
        grad : 4D np.array
           Gradient to be cropped.
            
        Returns
        -------
        4D np.array
            Cropped gradient
        """
        padding = self.hyperparameters['padding']
        
        return grad[padding:-padding, padding:-padding, :, :]