# -*- coding: utf-8 -*-
"""
@author: Prince Okoli

A multilayer percetron network for binary classification.

The nonflexible components are:
    Cost function: Cross entropy loss (binary).
    Optimizer: Vanilla gradient descent.

"""
import numpy as np
np.seterr(over="warn", under="warn") # warn for overflows and underflows.

# import custom modules.

from . import layer
from . import archiver
from . import _parameter
from . import _helper_funcs

# Check whether major version has changed, 
# and print warning to console if necessary.
_helper_funcs.check_numpy_ver()

class MLPNetwork: # public.
    """
    Standard feedforward neural network (a.k.a. multilayer percetron).
    
    DATA MEMBERS
    ------------
    X : TYPE numpy array. Must have the shape n x m, 
        where n is number of features and m is number of records.

    Y : TYPE numpy array. Must have the shape 1 x m, 
        where m is number of records.
    
    input_layer: TYPE _InputLayer (parent: Layer).
        
    
    training_archiver: TYPE TrainingArchiver.
        

    # TODO: Finish up list of attributes.
    """
    def __init__(self):
        self.X=None
        self.Y=None
        
        self.input_layer=layer._InputLayer(parent_network=self)
        
        self.layers=dict()
        
        self.temp_cache_network = dict()

        self.num_layers = None # excludes input layer in the count.
        
        self.Y_batch = None
        # input_layer handles X_batch. See the train method.
        
        self._add_input_layer()
        
        self.cost=None
        
        self.latest_accuracy=None # latest accuracy computed for a forward propagation.
        self.latest_precision=None # latest precision computed for a forward propagation.
        
        self.Y_pred=None # latest Y_pred computed for the latest forward propagation.
        
        self.num_latest_iteration=0
        
        self.parameter_initializer=None
        self.training_archiver=None
        
    def add_layers(self, layers): # public.
        """
        Adds one or more layers to the network.

        PARAMETERS
        ----------
        layers: TYPE Layer. Layer or collection of layers to be added.

        RETURNS
        -------
        None.

        """
        if isinstance(layers, layer.Layer):
            # if just one layer.
            self._add_layer(layers)
        if _helper_funcs.is_iterable(layers):
            # if a collection of layer(s).
            for l in layers:
                self._add_layer(l)
            
            
    def add_training_archiver(self, training_archiver): # public.
        self.training_archiver=training_archiver
        training_archiver._set_target_network(self)
        
    
    def initialize_parameters(self, weight_init_scheme="xavier", bias_init_scheme="zeros", 
                              factor=0.01,random_seed=3): # public.
        """
        
        Randomly initializes the weights according to the specified schemes, and the biases to zero. 
        
        PARAMETERS
        ----------
                    
        weight_init_scheme: TYPE Str. Case insensitive.
            The scheme used to initialize the weights.
            Options are:
            
                "default": initializes the weights using the standard normal distribution and rescales according to the specified factor.
                
                "xavier": scales the weights by a factor of 1/sqrt(n), where n is number of nodes in preceding layer.
                    
        bias_init_scheme: TYPE Str. Case insensitive.
            The scheme used to initialize the biases. 
            Options are:
                
                "ones": initializes bias to ones and rescales according to the specified factor.
                
                "zeros": initializes bias to zeros and rescales according to the specified factor.
            
        factor: TYPE numerical. Range is from 0 to 1. Default is 0.01.
            Rescales all parameters by a specified factor during initialization. 
            
        random_seed: TYPE Int.
            seed for pseudo random number generator.

        Returns
        -------
        None.

        """
        if (factor<=0 or factor>=1):
            raise ValueError("factor must range from 0 to 1.")
            
        weight_init_scheme = weight_init_scheme.lower()
        bias_init_scheme = bias_init_scheme.lower()
                
        self.parameter_initializer=_parameter.ParameterInitializer(weight_init_scheme=weight_init_scheme, 
                                                         bias_init_scheme=bias_init_scheme,
                                                         factor=factor,random_seed=random_seed)
        self.parameter_initializer.parent_network=self

    def _is_initializer_added(self): # module-private.
        if (self.parameter_initializer==None):
            return False
        else:
            return True
    # public.
    def train(self, X, Y, num_iterations=10, batch_size=None, learning_rate=0.0000009, print_start_end=True,
              validation_X=None, validation_Y=None): 
        
        self._check_readiness_to_train() # Raises an exception if not ready.
        
        if (Y.shape[1]!=X.shape[1] or Y.shape[0]!=1):
            raise ValueError("X and Y must have compatible shapes, n x m and 1 x m respectively.")
            
        if (Y.shape[0] != self.layers[self.num_layers].num_units):
            raise ValueError("Y and the output layer must have compatible shapes.")
        
        self.Y=Y 
        self.X=X         
        
        if print_start_end==True: print("Training Begins...")

        num_iterations=num_iterations+self.num_latest_iteration
        
        for self.num_latest_iteration in range(self.num_latest_iteration+1, num_iterations+1):
            # loop header: allows training to resume from the previous state of the network
            #  at end of its last training if any.
                
            # select batch from the training dataset.
            if batch_size is None: batch_size=self.X.shape[1] # use the entire data.
            
            random_indices = np.random.choice(self.Y.shape[1], (batch_size,), replace=False)
            
            self.Y_batch=self.Y[:,random_indices]
            self.input_layer._populate(self.X[:,random_indices])
                  
            # Forward propagation.
            self._network_forward_prop()
        
            # Back propagation.
            self._network_back_prop()
         
            # Update parameters.
            self._update_parameters_gradient_descent(learning_rate=learning_rate)
            
            # Compute the training cost, accuracy and precision (using the current training batch).
            
            self.training_archiver._compute_and_archive_cost(cost_type="training")
            self.training_archiver._compute_and_archive_accuracy(acc_type="training")
            self.training_archiver._compute_and_archive_precision(precis_type="training")
            
            # Compute the validation cost, accuracy and precision (using validation dataset).
            if (validation_Y is None) or (validation_X is None):
                pass
            else:
                self.input_layer._populate(validation_X)
                self.Y_batch = validation_Y
                
                self._network_forward_prop()
                self.training_archiver._compute_and_archive_cost(cost_type="validation")
                self.training_archiver._compute_and_archive_accuracy(acc_type="validation")
                self.training_archiver._compute_and_archive_precision(precis_type="validation")
            
            # rest of caching occurs here.
            
            self.training_archiver._archive_gradients()
            self.training_archiver._archive_parameters()
            
            # print archiving messages if any:
            if self.training_archiver.report: 
                self.training_archiver._print_report()
                self.training_archiver._clear_report()
            
        if print_start_end: print("Training Complete!")
            
    def evaluate(self, X, Y, metric="accuracy"): # public.
        # assumes binary classification.
        
        _available_perfomance_metrics=["accuracy","precision"]
        
        metric=metric.lower()
        
        if not any(m == metric.lower() for m in _available_perfomance_metrics):
            raise ValueError
                
        self.input_layer._populate(X)
        self.Y_batch=Y
        
        self._network_forward_prop()
                
        if metric=="accuracy":
            self._compute_accuracy()
            score=self.latest_accuracy
        if metric =="precision":
            self._compute_precision()
            score=self.latest_precision
            
        return score
        
    def predict(self, X): #public.
        self.input_layer._populate(X)
        self._network_forward_prop()
        return self.Y_pred
        
    def _check_readiness_to_train(self):
        """
        Check that at least one layer (excluding input layer) has been added, 
        parameters of all layers have been initialized, and a training archiver
        has been added.
        
        RAISES
        ------
        ValueError:
            If any of the checks fail.

        Returns
        -------
        None.

        """
        
        if self.num_layers is None or self.num_layers == 0:
            raise ValueError("No layers have been added to the network.")
        
        for l in self.layers.keys():
            if l==0:
                continue # no need to check input layer.
            if not self._is_initializer_added():
                raise ValueError("Parameters have not been initialized.")
        
        if not isinstance(self.training_archiver, archiver.TrainingArchiver):
            raise ValueError("A training archiver to cache data from training and display results has not been added.")
        
    def _update_num_layers(self): # class-private.
        self.num_layers=len(self.layers) - 1
        
        
    def _update_Y_pred(self): # class-private.
        L=self.num_layers
        self.Y_pred=np.where(self.layers[L].A>0.5, 1,0)
    
    
    def _add_input_layer(self): # class-private.
        self.layers[0]=self.input_layer
        self._update_num_layers()
    
    
    def _add_layer(self, layer): # class-private.
        """
        Adds one layer to the network. 
        Updates parent_network and position_in_network of the layer.
        
        PARAMETERS
        ----------
        layer: TYPE Layer. 
            The layer to be added.
        
        RETURNS
        -------
        None.

        """     
        if layer.parent_network==None:
            self.layers[self.num_layers+1]=layer
            self._update_num_layers()
            layer._incorporate_into_network(parent_network=self)
        else:
            raise ValueError("The layer has already been added to a network.")

    
    def _network_forward_prop(self): # class-private.
        """
        Performs forward propagation for the network.        

        Returns
        -------
        None.

        """
        self.parameter_initializer._execute_initialization_if_notdone()
            
        L = self.num_layers
    
        for l in range(1, L):
        # looping through all L-1 hidden layers.
            self.layers[l]._layer_forward_prop()            
                    
        else: # last layer.
            self.layers[L]._layer_forward_prop()
            
        self._update_Y_pred()
        
        return None    
    
    def _compute_cost(self): # module-private.
        """
        Computes cost using the cross-entropy cost function and assumes binary classification.
    
        Returns
        -------
        None.

        """
        L=self.num_layers
        m = self.Y_batch.shape[1] # number of records/instances/examples.
        A_last=self.layers[L].A
        
        # Computes cross entropy loss. The equation assumes both A_last and Y_batch are vectors (binary classification).
        self.cost = (-1/m) * np.sum((self.Y_batch * np.log(A_last)) + 
                                              ((1 - self.Y_batch) * np.log(1 - A_last)))
        
        self.cost = np.squeeze(self.cost) # ensures cost is a scalar (this turns [[10]] or [10] into 10).
            
        return None
    
    def _compute_last_layer_dJdA(self): # class-private.
        """
        Compute dJ/dA for the last layer. This assumes a cross-entropy cost function.
        """
        L=self.num_layers
        m = self.Y_batch.shape[1]
        A_last=self.layers[L].A
        
        self.layers[L].gradients["dJdA"] = -(1/m) * ((self.Y_batch / A_last) - 
                                    ((1 - self.Y_batch) / (1 - A_last)))
        return None

    def _network_back_prop(self): # class-private.
        """
        Performs backward propagation for the network.
        
        Returns
        -------
        None.

        """
        L = self.num_layers
        
        # Initialize the backpropagation by computing dJ/dA of the last layer (Lth layer).
        self._compute_last_layer_dJdA()
            
        # Compute the Lth layer gradients.
        last_layer = self.layers[L]
        last_layer._layer_back_prop()
        
        # ensure dJdB is a 2D numpy array and not 1D, even though it stored as a vector, 
        # and only broadcasted into a matrix during computations.
        last_layer.gradients["dJdB"] = last_layer.gradients["dJdB"].reshape(-1,1)
        
        # Compute the gradients of the other layers.
        for l in reversed(range(1, L)):
            current_layer = self.layers[l]
            
            current_layer._layer_back_prop()
                    
            # ensure dJdB is a 2D numpy array and not 1D.
            current_layer.gradients["dJdB"] = current_layer.gradients["dJdB"].reshape(-1,1)  
    
        return None
    
    def _update_parameters_gradient_descent(self, learning_rate): # class-private.
        """
        Update parameters of the network using standard gradient descent.        

        Parameters
        ----------
        learning_rate : TYPE numerical. 
            Learning rate controls the scale of the update. Ranges between 0 and 1.

        Returns
        -------
        None.

        """        
        L = self.num_layers # number of layers in the network (also sn of last layer).
    
        # the basic gradient descent.
        for l in range(1, L+1):
            self.layers[l].W = self.layers[l].W - learning_rate * self.layers[l].gradients["dJdW"]
            self.layers[l].B = self.layers[l].B - learning_rate * self.layers[l].gradients["dJdB"]        
        
        return None
    
    def _compute_accuracy(self): # module-private.
        Y_true=self.Y_batch.reshape(-1,)
        Y_pred=self.Y_pred.reshape(-1,)
        
        # assumes binary classification.
        self.latest_accuracy=np.average(np.where(Y_true==Y_pred, 1,0))
        
        return None
    
    def _compute_precision(self): # module-private.
        Y_true=self.Y_batch.reshape(-1,)
        Y_pred=self.Y_pred.reshape(-1,)
        
        # assumes binary classification.
        mask_pred_positives = (Y_pred==1)
        self.latest_precision=np.average(np.where(Y_pred[mask_pred_positives]==Y_true[mask_pred_positives], 1, 0))
        
        return None