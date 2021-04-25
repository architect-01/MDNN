import numpy as np

class Dense:
    """ Fully connected layer """
    
    def __init__(self, n_input_units, n_hidden_units, learning_rate = 1e-3):
        """Creates and initializes weights / biases of layer
        
        Parameters
        ----------
        n_input_units : int
            Number of input units (output units from the previous layer)
        n_hidden_units : int
            Number of hidden units in this layer
        
        Returns
        -------
        None
        """
        
        self.type = 'Dense'
        
        self.hyperparameters = {'learning_rate': learning_rate}
        
        self.shape = {'ni': n_input_units,
                      'no': n_hidden_units}
        
        self.parameters = {'W': self._xavier_init(), 
                           'b': np.zeros((self.shape['no'], 1))}
        
        self.cache = {'Z': None}
        
        
    def _xavier_init(self):
        """Initialize and return weights of the layer using Xavier's initialization
         
         Parameters
         ----------
        
         Returns
         -------
         np.array(n_hidden_units, n_input_units)
             Initialized weights.
        """
                
        sd = np.sqrt(2.0 / (self.shape['no'] + self.shape['ni'])) #standard deviation
        return np.random.uniform(low=-sd, high=sd, size=(self.shape['no'], self.shape['ni']))
    
    def forward_prop(self, Z):
        """Calculate forward propagation of this layer
        
        Parameters
        ----------
        Z : np.array(n_output_units_of_previous_layer, number_of_examples)
            Output of the previous layer.

        Returns
        -------
        np.array
            Calculated output
        """
        #Z is used in backward pass to update the weights
        self.cache = {'Z': Z}
    
        
        #calculate output of this layer
        O = np.dot(self.parameters['W'], Z) + self.parameters['b']

        return O
    
    def backward_prop(self, grad):
        """Calcutes the backward pass of this layer and updates it's parameters
        
        Parameters
        ----------
        grad : np.array
            Gradient comming from the previous layer (looking from the end to the start)
        
        Returns
        -------
        next_grad
            Gradient to be used in the next layer (looking from the end to the start)
        """
        
        _, m = grad.shape # m is the number of examples

        #updates (division by m is used to lower the possibility of overflow)
        dW = np.dot(grad / m, self.cache['Z'].T)
        db = np.sum(grad / m, axis = 1, keepdims = True)
        
            
        next_grad = np.dot(self.parameters['W'].T, grad)
        
        #parameters update using pure Gradient Descent, TODO: add other optimization methods
        self.parameters['W'] = self.parameters['W'] - self.hyperparameters['learning_rate'] * dW
        self.parameters['b'] = self.parameters['b'] - self.hyperparameters['learning_rate'] * db
        
        return next_grad