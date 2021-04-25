import numpy as np

class Activation:
    """Activation layer - to indtroduce non-linearities"""
    
    def __init__(self, func_name = 'relu'):
        """Assigns the corresponding function to the layer - both for the forward and backward prop
        
        Parameters
        ----------
        func_name : str
            Name of the activation function.
        
        Returns
        -------
        None
            a list of strings representing the header columns
        """
        
        self.type = f'Activation:{func_name}'
        
        self.cache = {'O': None}
        
        if func_name == 'relu':
            self.func = self._relu; self._der_func = self._der_relu
        elif func_name == 'sigmoid':
            self.func = self._sigmoid; self._der_func = self._der_sigmoid
        elif func_name == 'softmax':
            self.func = self._softmax; self._der_func = self._der_softmax
    
    def forward_prop(self, Z):
        """Computes forward propagation of the layer by applying the activation function elementwise
        
        Parameters
        ----------
        Z : np.array
            Output of the previous layer.
        
        Returns
        -------
        np.array
            Activated Z.
        """
        
        self.cache['O'] = self.func(Z)  # layer's output
        
        return self.cache['O'] 
    
    def backward_prop(self, grad):
        """Computes backward propagation of the layer
        
        Parameters
        ----------
        grad : np.array
            Gradient comming from the previous layer (looking from the model's end to it's start)
        
        Returns
        -------
        np.array
            Gradient for the next layer (looking from the model's end to it's start)
        """
        #softmax is a special case
        if self.func == self._softmax:
            return self._der_softmax(grad)
        
        next_grad = self._der_func() * grad #elementwise multiplication
        
        #print('relu next_grad.sum', next_grad.sum())
        print('grad.shape:', grad.shape)
        return next_grad
    
    def _relu(self, Z):        
        """Applies Rectified Linear Unit (ReLU) elementwise to Z
        
        Parameters
        ----------
        Z : np.array
            Output of the previous layer.
        
        Returns
        -------
        np.array
            Activated Z.
        """
        return np.maximum(Z, 0.0)
    
    def _der_relu(self):
        """Computes the derivative of the Rectified Linear Unit (ReLU) elementwise to Z
        
        Parameters
        ----------

        Returns
        -------
        np.array
            
        """
        return np.array(self.cache['O'] > 0, dtype = float)
    
    def _sigmoid(self, Z):
        """Applies Sigmoid function elementwise to Z
        
        Parameters
        ----------
        Z : np.array
            Output of the previous layer.
        
        Returns
        -------
        np.array
            Activated Z.
        """
        return 1.0 / (1.0 + np.exp(-Z)) 
    
    def _der_sigmoid(self):
        """Computes the derivative of Sigmoid elementwise to Z
        
        Parameters
        ----------

        Returns
        -------
        np.array

        """
        o = self.cache['O']
        
        return o * (1.0 - o)
    
    def _softmax(self, Z):
        """Applies Softmax function to Z
        
        Parameters
        ----------
        Z : np.array
            Output of the previous layer.
        
        Returns
        -------
        np.array
            Activated Z.
        """
        e_x = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
        return e_x / e_x.sum(axis = 0, keepdims = True)
    

    def _der_softmax(self, grad):
        """Computes the gradient to be passed to the next layer (end->start)
        
        Parameters
        ----------
        grad: np.array
            Gradient comming from the previous layer.    
        
        Returns
        -------
        np.array
            Gradient going to the next layer.
        """
        
        n_classes, m = grad.shape
    
        q = np.repeat(np.reshape(self.cache['O'], (n_classes, 1, m)), n_classes, 1)
        
        e = np.repeat(np.reshape(np.eye(n_classes), (n_classes, n_classes, 1)), m, 2)
        
        t = e - q
        

        next_grad = []
        for mi in range(m): #for every example compute the change
            grad_mi = np.repeat(grad[:, mi:mi+1], n_classes, 1)
            
            o_mi = np.repeat(self.cache['O'][:, mi:mi+1], n_classes, 1)
        
            change = t[:, :, mi].T * grad_mi * o_mi
            
            next_grad.append(change.sum(axis = 0, keepdims = True).tolist())
            
            #print(next_grad[-1].shape)
        
        next_grad = np.array(next_grad).reshape(m, n_classes).T
        
        #print(next_grad[:, 1])
        
        return next_grad