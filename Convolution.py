import numpy as np

class Convolution:
    """ Fully connected layer """
    
    def __init__(self, image_shape, kernel_size, n_kernel, step_size = 3, learning_rate = 1e-2):
        """Creates and initializes weights / biases of layer
        
        Parameters
        ----------
        kernel_size : 3D tupple
            height, width of the kernel and number of channels
        n_kernel : int
            Number of kernels in this layer
        
        Returns
        -------
        None
        """
        
        self.type = 'Convolution'
        
        #image height and width
        i_h, i_w, n_ch = image_shape
        
        k_h, k_w, k_ch = kernel_size
        
        #calculate the number of sections in the image
        n_passes_h, n_passes_w = 1 + (i_h - k_h) // step_size, 1 + (i_w - k_w) // step_size
        
        self.hyperparameters = {'learning_rate': learning_rate,
                                'kernel_size': kernel_size,
                                'n_kernel': n_kernel,
                                'step_size': step_size}
        
        #weights of kernels are organized in a matrix of shape (number of neurons in kernel, number of kernels)
        self.parameters = {'W': self._xavier_init(), 
                           'b': np.zeros((n_passes_h, n_passes_w, n_kernel, 1))}
        
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
        kernel_size = self.hyperparameters['kernel_size']
        n_kernel = self.hyperparameters['n_kernel']
        n_neurons_in_kernel = kernel_size[0]*kernel_size[1]*kernel_size[2]
                
        sd = np.sqrt(2.0 / n_neurons_in_kernel) #standard deviation
        return np.random.uniform(low=-sd, high=sd, size=(n_kernel, n_neurons_in_kernel))
    
    def forward_prop(self, Z):
        """Calculate forward propagation of this layer
        
        Parameters
        ----------
        Z : np.array(height, width, number_of_examples)
            Output of the previous layer.

        Returns
        -------
        np.array
            Calculated output
        """
        #image height and width
        i_h, i_w, n_ch, _ = Z.shape
        
        k_h, k_w, k_ch = self.hyperparameters['kernel_size'] #kernel's height and width
        
        step_size = self.hyperparameters['step_size']
        
        n_kernel = self.hyperparameters['n_kernel']
        
        #calculate the number of sections in the image
        n_passes_h, n_passes_w = 1 + (i_h - k_h) // step_size, 1 + (i_w - k_w) // step_size
                
        #print(f'n_passes_h: {n_passes_h}, n_passes_w: {n_passes_w}')
        
        #padd with zeros to account for the remainder of images
        padded_h, padded_w = - i_h + n_passes_h*step_size + k_h, - i_w + n_passes_w*step_size + k_w
        
        #print(f'padded_h: {padded_h}, padded_w: {padded_w}')

    
        #Z is used in backward pass to update the weights
        self.cache = {'Z': np.pad(Z, ((0, padded_h),
                                      (0, padded_w), 
                                      (0, 0),
                                      (0, 0)),
                                  mode='constant'),
                      'padded_h' : padded_h,
                      'padded_w' : padded_w} #need to store the padded amounts - in the back_prop it will be discarded
        
        
        self.cache['image_shape'] = self.cache['Z'].shape[:-1]
    
    
        #section the images
        sectioned_Z = self.cache['sectioned_Z'] = self._slice_the_images_and_flatten_the_sections()
        #print(f'sectioned_Z : {sectioned_Z.shape}')
        
        
        #calculate output of this layer
        O = []
        n_sections, _, m = sectioned_Z.shape

        for sect_i in range(n_sections):
            O.append(np.dot(self.parameters['W'], sectioned_Z[sect_i]))
            
        O = np.array(O).reshape((n_passes_h, n_passes_w, n_kernel, -1)) + self.parameters['b']
        #print(f'Conv layer output shape: {O.shape}')

        return O
    

    
    def _slice_the_images_and_flatten_the_sections(self):
        """Slices and flattens the sections of input images such that they can be used to compute the forward propagation
        
        Parameters
        ----------
            
        Returns
        -------
        3D np.array
            Sliced and flatten sections
        """
        
        
        images = self.cache['Z'] #note: this may not be images but feature maps of the previous conv. layer
        
        i_h, i_w, n_ch = self.cache['image_shape'] ;
        
        k_h, k_w, k_ch = self.hyperparameters['kernel_size'] #kernel's height, width and number of channels
        
        step_size = self.hyperparameters['step_size']
        
        #print(f'image shape : {i_h},{i_w}')
        
        #print(f'(i_h-k_h)/step_size = {(i_h-k_h)/step_size}')
        #print(f'(i_w-k_w)/step_size = {(i_w-k_w)/step_size}')
                
        #store the sections of images that will be used to create feature maps
        sections = []
        for y_start in range(0, i_h - k_h, step_size):
            
            #print(f'y_start: {y_start}')
            
            y_end = y_start + k_h 
            
            for x_start in range(0, i_w - k_w, step_size):
                
                x_end = x_start + k_w 
                
                image_section = images[y_start:y_end, x_start:x_end, :, :]
                
                #print(f'image_section.shape : {image_section.shape}')
                
                sections.append(np.reshape(image_section, (k_h*k_w*k_ch, -1)))
        
        
        sections = np.array(sections)
        
        #print(f'sections shape = {sections.shape}')
        
        return sections
        
        
    def backward_prop(self, grad):
        """Computes backward propagation
        
        Parameters
        ----------
        grad : 4D np.array
            Gradient flowing from the higher layers
            
        Returns
        -------
        4D np.array
            Gradient going to the lower layers
        """
        
        sectioned_Z = self.cache['sectioned_Z']
        kernel_size = self.hyperparameters['kernel_size']
        step_size = self.hyperparameters['step_size']
        k_h, k_w, k_ch = kernel_size
        
        t0, t1, n_kernel, m = grad.shape
        
        next_grad = np.zeros(self.cache['Z'].shape)
        
        #parameters
        W = self.parameters['W']
        
        for kernel_i in range(n_kernel): #go through every kernel
            
            #used to compute the gradiend for the next layer
            repeated_reshaped_kernel = W[kernel_i:kernel_i+1, :].reshape(*kernel_size, 1).repeat(m, -1)

            #go through every gradient point
            for y_grad in range(t0):
                
                y1 = y_grad*step_size ; y2 = y1 + k_h
                
                for x_grad in range(t1):
                    
                    x1 = x_grad*step_size ; x2 = x1 + k_w
                    
                    change = repeated_reshaped_kernel*grad[y_grad:y_grad+1, x_grad:x_grad+1, kernel_i:kernel_i+1, :]
                    
                    next_grad[y1:y2, x1:x2, :, :] += change
                    
                    
            kernel_update = sectioned_Z * grad[:, :, kernel_i:kernel_i+1, :].reshape((t0*t1, 1, -1))
            #update the parameters
            W[kernel_i:kernel_i+1, :] -= self.hyperparameters['learning_rate'] * kernel_update.sum(axis=0).sum(axis=-1) / m
            
        #update bias
        self.parameters['b'] -= self.hyperparameters['learning_rate'] * grad.sum(axis=-1, keepdims=True) / m
            
        padded_h, padded_w = self.cache['padded_h'], self.cache['padded_w']
                
        next_grad = next_grad[:-padded_h, :-padded_w, :, :] #crop away the padding
        
        return next_grad