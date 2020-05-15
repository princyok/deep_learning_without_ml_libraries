# -*- coding: utf-8 -*-
"""
@author: Prince Okoli

A multilayer percetron network for binary classification.

The nonflexible components are:
    Cost function: Cross entropy loss (binary).
    Parameter optimizer: Vanilla gradient descent.
"""

import numpy as np
import copy
import helper_funcs
np.seterr(over="warn", under="warn") # warn for overflows and underflows.

# check whether major version has changed.

if(int(np.__version__[0])!=1):
    message="Caution: This module was created with numpy major version 1, "+\
    "but the current major version is "+str(np.__version__[0])
    print(message)
    
    
class Layer: # public.
    """
    Standard feedforward neural network (a.k.a. multilayer pecetron).
    
    DATA MEMBERS
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

    def _layer_forward_prop(self): # module-private.
        """
        Performs the forward propagation for a layer.
        """        
        self._compute_linear_preactivation()
        
        self.A=self.activation_type._forward_pass(self.Z)
        
        return None 
        
    def _incorporate_into_network(self, parent_network): # module-private.
            
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
    
    def _layer_back_prop(self): # module-private.
        """
        Peforms backward propagation for a layer.
        """
        Z = self.Z
        A = self.A
        
        self.gradients["dAdZ"]=self.activation_type._backward_pass(A, Z)

        self.gradients["dJdZ"] =self.gradients["dJdA"] * self.gradients["dAdZ"]
        
        self._compute_cost_gradients()
        
        return None
                
    def _compute_linear_preactivation(self): # class-private.
        """
        Computes the preactivation for a layer's forward propagation.
        
        """
        A_prior = self.preceding_layer.A

        self.Z = np.matmul(self.W, A_prior) + self.B
                    
        return None
    
class _InputLayer(Layer): # module-private.
    def __init__(self, parent_network):
        super().__init__(activation_name=None, num_units= None)
        self.position_in_network=0
        self.parent_network=parent_network
    def _populate(self, X):
        self.A=X
        self.num_units=X.shape[0]

class Network: # public.
    """
    Standard feedforward neural network (a.k.a. multilayer pecetron).
    
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
        
        self.input_layer=_InputLayer(parent_network=self)
        
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
        if isinstance(layers, Layer):
            # if just one layer.
            self._add_layer(layers)
        if helper_funcs.is_iterable(layers):
            # if a collection of layer(s).
            for layer in layers:
                self._add_layer(layer)
            
            
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
                
        self.parameter_initializer=_ParameterInitializer(weight_init_scheme=weight_init_scheme, 
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
        
        if not isinstance(self.training_archiver, TrainingArchiver):
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
        L = self.num_layers # number of layers in the network.
    
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
    
class _ParameterInitializer: # module-private.

    def __init__(self, weight_init_scheme="xavier", bias_init_scheme="zeros", 
                 factor=0.01,random_seed=3):
        
        self.weight_init_scheme=weight_init_scheme
        self.bias_init_scheme=bias_init_scheme
        self.factor=factor
        self.rnd=np.random.RandomState(random_seed)
        self.status=0
        self.parent_network=None
        
    def _execute_initialization_if_notdone(self):
        
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
    
class _ActivationFunction: # module-private.
    
    _available_activation_funcs=["logistic","relu","tanh"]
    
    def __init__(self, name):      
        if any(a == name.lower() for a in self.__class__._available_activation_funcs):
            self.name=name.lower()
        else:
            raise ValueError
        
    def _forward_pass(self, Z): # module-private.
        if self.name=="logistic":
            A = self._logistic(Z)
        if self.name=="relu":
            A = self._relu(Z)
        if self.name=="tanh":
            A = self._tanh(Z) 
        return A
    
    def _backward_pass(self, A,Z): # module-private.
        if self.name=="logistic":
            dAdZ=self._logistic_gradient(A)
        if self.name=="relu":
            dAdZ=self._relu_gradient(Z)
        if self.name=="tanh":
            dAdZ=self._tanh_gradient(Z)          
        return dAdZ
    
    def _logistic(self, Z): # class-private.
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

    def _relu(self, Z): # class-private.
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
    
    
    def _tanh(self, Z): # class-private.
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
    
    def _relu_gradient(self, Z): # class-private.
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
    
    def _logistic_gradient(self, A): # class-private.
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
    
    def _tanh_gradient(self, A): # class-private.
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
    
class TrainingArchiver:
    """
    An archiver that caches components and outputs of the network during training, like weights, gradients, accuracy, etc.
    The frequency of the archiving is user-specified.
    
    DATA MEMBERS
    ------------
    #TODO
        
    """
    archival_targets= ["activation", "preactivation", "cost", "gradient", 
                    "parameters", "accuracy", "precision"]
    num_archival_targets=len(archival_targets)
    
    def __init__(self, broad_frequency=None):

        self.archiving_frequencies=dict(
            zip(self.__class__.archival_targets,[broad_frequency]*self.__class__.num_archival_targets))
        
        self.archiving_verbosities=dict(
            zip(self.__class__.archival_targets,[0]*self.__class__.num_archival_targets))
        
        self.all_gradients=dict()
        self.all_parameters=dict()
        self.all_preactivations=dict()
        self.all_activations=dict()
        
        self.all_training_accuracies=dict()
        self.all_validation_accuracies=dict()
        self.all_training_precisions=dict()
        self.all_validation_precisions=dict()
        self.all_training_costs=dict()
        self.all_validation_costs=dict()
        
        self.target_network=None
        
        self.report=""
                
    def set_archiving_frequencies(self, **kwargs): # public.
        
        self.archiving_frequencies["activation"]=kwargs.get("activation", 0)
        self.archiving_frequencies["preactivation"]=kwargs.get("preactivation", 0)
        self.archiving_frequencies["cost"]=kwargs.get("cost", 0)
        self.archiving_frequencies["gradient"]=kwargs.get("gradient", 0)
        self.archiving_frequencies["parameters"]=kwargs.get("parameters", 0)
        self.archiving_frequencies["accuracy"]=kwargs.get("accuracy", 0)
        self.archiving_frequencies["precision"]=kwargs.get("precision", 0)
    
    def set_archiving_verbosities(self, **kwargs): # public.
        """
        Set whether the archiver will be verbose (displays summary) or silent while archiving.

        Parameters
        ----------
        kwargs** : TYPE Boolean, bit (e.g. 0 or 1), etc. Any value that resolves to Boolean.
            Valid arguments are: "activation", "preactivation", "cost", "gradient", 
                    "parameters", "accuracy", and "precision".
            True means summary of the archiving will be printed to console, and
            False means silent archiving.
        Returns
        -------
        None.

        """
        
        self.archiving_verbosities["activation"]=kwargs.get("activation", 0)
        self.archiving_verbosities["preactivation"]=kwargs.get("preactivation", 0)
        self.archiving_verbosities["cost"]=kwargs.get("cost", 0)
        self.archiving_verbosities["gradient"]=kwargs.get("gradient", 0)
        self.archiving_verbosities["parameters"]=kwargs.get("parameters", 0)
        self.archiving_verbosities["accuracy"]=kwargs.get("accuracy", 0)
        self.archiving_verbosities["precision"]=kwargs.get("precision", 0)
        
    def _set_target_network(self, target_network): # module-private.
        self.target_network=target_network
    
    def _archive_activations(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["activation"]!=0) and (i % self.archiving_frequencies["activation"] == 0):
            L = self.target_network.num_layers
            acts_all_layers=dict()
            for l in range(1, L+1):
                acts_all_layers[l]=copy.deepcopy(self.target_network.layers[l].A)
            self.all_activations[i]=acts_all_layers
    
    def _archive_preactivations(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["preactivation"]!=0) and (i % self.archiving_frequencies["preactivation"] == 0):
            L = self.target_network.num_layers
            preacts_all_layers=dict()
            for l in range(1, L+1):
                preacts_all_layers[l]=copy.deepcopy(self.target_network.layers[l].Z)
            self.all_preactivations[i]=preacts_all_layers        
        
    def _archive_gradients(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["gradient"]!=0) and (i % self.archiving_frequencies["gradient"] == 0):
            L = self.target_network.num_layers
            grads_all_layers=dict()
            for l in range(1, L+1):
                grads_all_layers[l]=copy.deepcopy(self.target_network.layers[l].gradients)
            self.all_gradients[i]=grads_all_layers
    
    def _archive_parameters(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["parameters"]!=0) and (i % self.archiving_frequencies["parameters"] == 0):
            L = self.target_network.num_layers
            params_all_layers=dict()
            for l in range (1, L+1):
                params_all_layers["W"+str(l)]=copy.deepcopy(self.target_network.layers[l].W)
                params_all_layers["B"+str(l)]=copy.deepcopy(self.target_network.layers[l].B)
            self.all_parameters[i]=params_all_layers
            
    def _compute_and_archive_accuracy(self, acc_type): # module-private.
        i = self.target_network.num_latest_iteration
        if (self.archiving_frequencies["accuracy"]!=0) and (i % self.archiving_frequencies["accuracy"] == 0):
            if acc_type=="training":
                self.target_network._compute_accuracy()
                self.all_training_accuracies[i]=self.target_network.latest_accuracy
                self._update_report(archival_target="accuracy", prefix="training")
            if acc_type=="validation":
                self.target_network._compute_accuracy()
                self.all_validation_accuracies[i]=self.target_network.latest_accuracy
                self._update_report(archival_target="accuracy", prefix="validation")
                
    def _compute_and_archive_precision(self, precis_type): # module-private.
        i = self.target_network.num_latest_iteration
        if (self.archiving_frequencies["precision"]!=0) and (i % self.archiving_frequencies["precision"] == 0):
            if precis_type=="training":
                self.target_network._compute_precision()
                self.all_training_precisions[i]=self.target_network.latest_precision
                self._update_report(archival_target="precision", prefix="training")
            if precis_type=="validation":
                self.target_network._compute_precision()
                self.all_validation_precisions[i]=self.target_network.latest_precision
                self._update_report(archival_target="precision", prefix="validation")
                
    def _compute_and_archive_cost(self, cost_type): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["cost"]!=0) and (i % self.archiving_frequencies["cost"] == 0):
            
            if cost_type=="training":
                self.target_network._compute_cost()
                self.all_training_costs[i] = self.target_network.cost
                self._update_report(archival_target="cost", prefix="training")
                
            if cost_type=="validation":
                self.target_network._compute_cost()
                self.all_validation_costs[i] = self.target_network.cost
                
                self._update_report(archival_target="cost", prefix="validation")

    def _update_report(self, archival_target, prefix=None, suffix=None): # module-private.
        if prefix==None: prefix=""
        if suffix==None: suffix=""
        
        i = self.target_network.num_latest_iteration
        
        if self.archiving_verbosities[archival_target]:
            
            if archival_target=="accuracy" and prefix=="validation":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_validation_accuracies[i]))+"\n"
            elif archival_target=="accuracy" and prefix=="training":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_training_accuracies[i]))+"\n"                
            
            if archival_target=="cost" and prefix=="validation":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_validation_costs[i]))+"\n"
            elif archival_target=="cost" and prefix=="training":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_training_costs[i]))+"\n"

            if archival_target=="precision" and prefix=="validation":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_validation_precisions[i]))+"\n"
            elif archival_target=="precision" and prefix=="training":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(helper_funcs.sigfig(self.all_training_precisions[i]))+"\n"
    def _clear_report(self):
        self.report=""
    def _print_report(self):
        print(self.report,"="*10)