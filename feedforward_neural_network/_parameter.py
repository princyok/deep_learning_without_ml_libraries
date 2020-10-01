# -*- coding: utf-8 -*-
"""
@author: Prince Okoli

Module for the weights and biases of all layers of the network.

"""
import numpy as np
np.seterr(over="warn", under="warn") # warn for overflows and underflows.

# import custom modules.
from . import _helper_funcs

# Check whether major version has changed, 
# and print warning to console if necessary.
_helper_funcs.check_numpy_ver()

class ParameterInitializer:

    def __init__(self, weight_init_scheme="xavier", bias_init_scheme="zeros", 
                 factor=0.01,random_seed=3):
        
        self.weight_init_scheme=weight_init_scheme
        self.bias_init_scheme=bias_init_scheme
        self.factor=factor
        self.rnd=np.random.RandomState(random_seed)
        self.status=0
        self.parent_network=None
        
    def execute_initialization_if_notdone(self):
        
        if self.status==0:
            L = self.parent_network.num_layers # number of layers in the network.
        
            for l in range(1, L+1):
                
                # initializing weights.
                self._initialize_weights(l)
                
                # initializing biases.
                self._initialize_biases(l)
                
            self.status=1
        
        return None
    
    def _initialize_weights(self, layer_sn):
        l=layer_sn
        if self.weight_init_scheme=="default":
            self.parent_network.layers[l].W = self.rnd.randn(self.parent_network.layers[l].num_units, 
                                              self.parent_network.layers[l-1].num_units)\
                * self.factor
        
        elif self.weight_init_scheme=="xavier":
            self.parent_network.layers[l].W = self.rnd.randn(self.parent_network.layers[l].num_units, 
                                              self.parent_network.layers[l-1].num_units)\
                / np.sqrt(self.parent_network.layers[l-1].num_units)
                
        return None
        
    def _initialize_biases(self, layer_sn):
        l=layer_sn
        if self.bias_init_scheme=="zeros":
            self.parent_network.layers[l].B = np.zeros((self.parent_network.layers[l].num_units, 1))\
                * self.factor
        elif self.bias_init_scheme=="ones":
            self.parent_network.layers[l].B = np.ones((self.parent_network.layers[l].num_units, 1))\
                * self.factor              

        return None
    