# -*- coding: utf-8 -*-
"""
@author: Prince Okoli

Module for the feedforward layer. Also includes functionalities for activation functions.

"""
import numpy as np
np.seterr(over="warn", under="warn") # warn for overflows and underflows.

# import custom modules.

from . import _helper_funcs

# Check whether major version has changed, 
# and print warning to console if necessary.
_helper_funcs.check_numpy_ver()

class Layer: # public.
    """
    Standard feedforward layer.
    
    ATTRIBUTES
    ------------
    num_units : TYPE numerical.
        Number of units in the network.

    W: TYPE numpy array.
        Weights of the layer.
        
        Has the shape n_current x n_previous, where n_current and n_previous 
        are number of units in current and previous layers respectively.
    
    B: TYPE numpy array.
        Biases of the layer.
        
        Has the shape n x 1, where n is the number of units in the layers.
         
    # TODO: Finish up list of attributes.
    """    
    def __init__(self, activation_name, num_units):
        
        if activation_name == None:
            self.activation_type=None            
        else:
            self.activation_type=_ActivationFunction(name=activation_name)
            
        self.num_units=num_units
        self.W=None
        self.B=None
        self.A=None
        self.Z=None
        self.gradients=dict(zip(["dAdZ", "dJdA", "dJdZ", "dJdW", "dJdB"],[None]*5))
        self.parent_network=None
        self.position_in_network=None
        self.preceding_layer=None

    def layer_forward_prop(self): # module-private.
        """
        Performs the forward propagation for a layer.
        """        
        self._compute_linear_preactivation()
        
        self.A=self.activation_type.forward_pass(self.Z)
        
        return None 
        
    def incorporate_into_network(self, parent_network): # module-private.
            
        self.parent_network=parent_network

        # assumes sequential adding of layers. E.g. Layer3 won't be added before Layer2.
        self.position_in_network=self.parent_network.num_layers
        
        self.preceding_layer=self.parent_network.layers[self.position_in_network - 1]
        
    def _compute_cost_gradients(self): # class-private.
        """
        Computes dJ/dW and dJ/dB of the layer and dJ/dA of the preceding layer.
        """
        
        A_prior = self.preceding_layer.A

        self.gradients["dJdB"] = np.sum(self.gradients["dJdZ"], axis=1) 
        
        self.gradients["dJdW"] = np.matmul(self.gradients["dJdZ"], A_prior.T)
        
        self.preceding_layer.gradients["dJdA"] = np.matmul(self.W.T, self.gradients["dJdZ"])
        
        return None
    
    def layer_back_prop(self): # module-private.
        """
        Peforms backward propagation for a layer.
        """
        Z = self.Z
        A = self.A
        
        self.gradients["dAdZ"]=self.activation_type.backward_pass(A, Z)

        self.gradients["dJdZ"] =self.gradients["dJdA"] * self.gradients["dAdZ"]
        
        self._compute_cost_gradients()
        
        return None
                
    def _compute_linear_preactivation(self):
        """
        Computes the preactivation for a layer's forward propagation.
        
        """
        A_prior = self.preceding_layer.A

        self.Z = np.matmul(self.W, A_prior) + self.B
                    
        return None
    
class _InputLayer(Layer):
    def __init__(self, parent_network):
        super().__init__(activation_name=None, num_units= None)
        self.position_in_network=0
        self.parent_network=parent_network
    def _populate(self, X):
        self.A=X
        self.num_units=X.shape[0]

class _ActivationFunction:
    
    _available_activation_funcs=["logistic","relu","tanh", "linear"]
    
    def __init__(self, name):      
        if any(a == name.lower() for a in self.__class__._available_activation_funcs):
            self.name=name.lower()
        else:
            raise ValueError
        
    def forward_pass(self, Z): # module-private.
        if self.name=="logistic":
            A = self._logistic(Z)
        if self.name=="relu":
            A = self._relu(Z)
        if self.name=="tanh":
            A = self._tanh(Z)
        if self.name=="linear":
            A = self._linear(Z)
        return A
    
    def backward_pass(self, A, Z): # module-private.
        if self.name=="logistic":
            dAdZ=self._logistic_gradient(A)
        if self.name=="relu":
            dAdZ=self._relu_gradient(Z)
        if self.name=="tanh":
            dAdZ=self._tanh_gradient(Z)
        if self.name=="linear":
            dAdZ = self._linear_gradient(Z)    
        return dAdZ
    
    def _logistic(self, Z):
        """
        The logistic activation function that maps preactivation to activation.
        
        Parameters
        ----------
        Z: TYPE numpy array.
            represents the preactivation.

        Returns
        -------
        A: TYPE numpy array.
            activation value, same shape as Z.

        """
    
        A = 1/(1+np.exp(-Z))
    
        return A

    def _relu(self, Z):
        """
        The rectified linear activation function that maps preactivation to activation.
    
        Parameters
        ----------
        Z: TYPE numpy array.
            represents the preactivation.

        Returns
        -------
        A: TYPE numpy array.
            activation value, same shape as Z.
        """
        A = np.maximum(0,Z)
    
        return A
    
    
    def _tanh(self, Z):
        """
        The hyperbolic tangent activation function that maps preactivation to activation.
    
        Parameters
        ----------
        Z: TYPE numpy array.
            represents the preactivation.

        Returns
        -------
        A: TYPE numpy array.
            activation value, same shape as Z.
        """
        A=(np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        return A
    
    def _linear(self, Z):
        return Z
    
    def _relu_gradient(self, Z):
        """
        Computes the gradient of a RELU unit w.r.t. the preactivation, for back propagation.

        Parameters
        ----------
        Z: TYPE numpy array.
            represents the preactivation.

        Returns
        -------
        dAdZ: TYPE numpy array.
            Gradient of the cost w.r.t. preactivation (dA/dZ).
        """
        result = np.array(Z, copy=True)
    
        # When z <= 0, the derivative is set to 0. Otherwise, set to 1. 
        result[Z <= 0] = 0
        result[Z > 0] = 1
        
        dAdZ=result
        return dAdZ
    
    def _logistic_gradient(self, A):
        """
        Computes the gradient for a logistic unit w.r.t. the preactivation, for back propagation.
        
        Parameters
        ----------
        A: TYPE numpy array.
            represents the activation.

        Returns
        -------
        dAdZ: TYPE numpy array.
            Gradient of the cost w.r.t. preactivation (dA/dZ).
        """
        dAdZ = A * (1-A)
    
        return dAdZ
    
    def _tanh_gradient(self, A):
        """
        Computes the gradient for a hyperbolic tangent unit w.r.t. the preactivation, for back propagation.        
        
        Parameters
        ----------
        A: TYPE numpy array.
            represents the activation.

        Returns
        -------
        dAdZ: TYPE numpy array.
            Gradient of the cost w.r.t. preactivation (dA/dZ).
        """
        dAdZ=1-A**2
        
        return dAdZ
        
    def _linear_gradient(self, Z):
        return np.ones(Z.shape)
    