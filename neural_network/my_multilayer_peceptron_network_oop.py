# -*- coding: utf-8 -*-
"""
@author: Prince Okoli

A multilayer percetron network for binary classification.

The nonflexible components are:
    Cost function: Cross entropy loss.
    Parameter optimizer: Vanilla gradient descent.
"""

import numpy as np
import copy
import helper_funcs
np.seterr(over="warn", under="warn") # warn for overflows and underflows.

class Layer:
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
            self.activation_type=ActivationFunction(name=activation_name)
            
        self.num_units=num_units
        self.W=None
        self.B=None
        self.A=None
        self.Z=None
        self.gradients=dict(zip(["dAdZ", "dJdA", "dJdZ", "dJdW", "dJdB"],[None]*5))
        self.parent_network=None
        self.position_in_network=None
        self.preceding_layer=None
        
    def incorporate_into_network(self, parent_network):
            
        self.parent_network=parent_network

        # assumes sequential adding of layers. E.g. Layer3 won't be added before Layer2.
        self.position_in_network=self.parent_network.num_layers
        
        self.preceding_layer=self.parent_network.layers[self.position_in_network - 1]
            
    def compute_linear_preactivation(self): 
        """
        Computes the preactivation for a layer's forward propagation.
        
        """
        A_prior = self.preceding_layer.A

        self.Z = np.matmul(self.W, A_prior) + self.B
                    
        return None

    def layer_forward_prop(self):
        """
        Performs the forward propagation for a layer.
        """        
        self.compute_linear_preactivation()
        
        self.A=self.activation_type.forward_pass(self.Z)
        
        return None 

    def compute_cost_gradients(self):
        """
        Computes dJ/dW and dJ/dB of the layer and dJ/dA of the preceding layer.
        """
        
        
        A_prior = self.preceding_layer.A
        
        m = A_prior.shape[1] # activation is expected to have shape of the form (# of units in layer, # of examples in batch).
        
        # for training batch of size > 1,  a compromise for dJdB, is an average of dJdZ across the batch.
        # using np.sum instead of np.average for readability.
        self.gradients["dJdB"] = (1/m) * np.sum(self.gradients["dJdZ"], axis=1) 
        
        # rescale by same factor as dJdB, to keep scaling consistent with dJdB.
        self.gradients["dJdW"] = (1/m) * np.matmul(self.gradients["dJdZ"], A_prior.T)
        
        # don't scale dJdA.
        self.preceding_layer.gradients["dJdA"] = np.matmul(self.W.T, self.gradients["dJdZ"])
        
        return None
    
    def layer_back_prop(self):
        """
        Peforms backward propagation for a layer.
        """
        Z = self.Z
        A = self.A
        
        self.gradients["dAdZ"]=self.activation_type.backward_pass(A, Z)

        self.gradients["dJdZ"] = self.gradients["dAdZ"] * self.gradients["dJdA"]
        
        self.compute_cost_gradients()
        
        return None

class InputLayer(Layer):

    def __init__(self, X, parent_network):
        
        super().__init__(activation_name=None, num_units= X.shape[0])
        self.A=X
        self.position_in_network=0
        self.parent_network=parent_network

class Network:
    """
    Standard feedforward neural network (a.k.a. multilayer pecetron).
    
    DATA MEMBERS
    ------------
    X : TYPE numpy array. Must have the shape n x m, 
        where n is number of features and m is number of records.

    Y : TYPE numpy array. Must have the shape 1 x m, 
        where m is number of records.
    
    input_layer: TYPE InputLayer (parent: Layer).
        
    
    training_archiver: TYPE TrainingArchiver.
        

    # TODO: Finish up list of attributes.
    """
    def __init__(self, X, Y):
        
        if (Y.shape[1]!=X.shape[1] or Y.shape[0]!=1):
            raise ValueError("X and Y must have compatible shapes, n x m and 1 x m respectively.")
        self.Y=Y
        self.X=X
        
        self.input_layer=InputLayer(X=self.X, parent_network=self)
        
        self.cost=None
        self.layers=dict()
        
        self.temp_cache_network = dict()

        self.num_layers = None # excludes input layer in the count.
        
        self.Y_batch = None
        # input_layer handles X_batch. See the train method.
        
        self._add_input_layer()
        
        self.latest_accuracy=None # latest accuracy computed for a forward propagation.
        self.latest_precision=None # latest precision computed for a forward propagation.
        
        self.Y_pred=None # latest Y_pred computed for the latest forward propagation.
        
        self.num_latest_iteration=0
        
        self.training_archiver=None
        
    def add_training_archiver(self, training_archiver):
        self.training_archiver=training_archiver
        training_archiver.set_target_network(self)
        
    def _update_num_layers(self):
        self.num_layers=len(self.layers) - 1
        
    def _update_Y_pred(self):
        L=self.num_layers
        self.Y_pred=np.where(self.layers[L].A>0.5, 1,0)
    
    def _add_input_layer(self):
        self.layers[0]=self.input_layer
        self._update_num_layers()
    
    def add_layers(self, layers):
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
    
    def _add_layer(self, layer):
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
            layer.incorporate_into_network(parent_network=self)
        else:
            raise ValueError("The layer has already been added to a network.")
    
    def initialize_parameters(self, weight_init_scheme="xavier", bias_init_scheme="zeros", factor=0.01,random_seed=3):
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
        weight_init_scheme = weight_init_scheme.lower()
        bias_init_scheme = bias_init_scheme.lower()
        
        assert (factor>=0 and factor<=1) # factor must range from 0 to 1.
        
        rnd=np.random.RandomState(random_seed)
        
        L = self.num_layers # number of layers in the network.
    
        for l in range(1, L+1):
            
            # initializing weights.

            if weight_init_scheme=="default":
                self.layers[l].W = rnd.randn(self.layers[l].num_units, self.layers[l-1].num_units) * factor
            elif weight_init_scheme=="xavier":
                self.layers[l].W = rnd.randn(self.layers[l].num_units, self.layers[l-1].num_units) / np.sqrt(self.layers[l-1].num_units)
            
            # initializing biases.
            
            if bias_init_scheme=="zeros":
                self.layers[l].B = np.zeros((self.layers[l].num_units, 1)) * factor
            elif bias_init_scheme=="ones":
                self.layers[l].B = np.ones((self.layers[l].num_units, 1)) * factor 
        return None
    
    def network_forward_prop(self):
        """
        Performs forward propagation for the network.        

        Returns
        -------
        None.

        """
        L = self.num_layers
    
        for l in range(1, L):
        # looping through all L-1 hidden layers.
            self.layers[l].layer_forward_prop()            
                    
        else: # last layer.
            self.layers[L].layer_forward_prop()
            
        self._update_Y_pred()
        
        return None
    
    def compute_cost(self):
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
        self.cost = (-1 / m) * np.sum((self.Y_batch * np.log(A_last)) + 
                                              ((1 - self.Y_batch) * np.log(1 - A_last)))
        
        self.cost = np.squeeze(self.cost) # ensures cost is a scalar (this turns [[10]] or [10] into 10).
            
        return None

    def compute_last_layer_dJdA(self):
        """
        Compute dJ/dA for the last layer. This assumes a cross-entropy cost function.
        """
        L=self.num_layers
        A_last=self.layers[L].A
        
        self.layers[L].gradients["dJdA"] = - ((self.Y_batch / A_last) - 
                                    ((1 - self.Y_batch) / (1 - A_last)))
        return None

    def network_back_prop(self):
        """
        Performs backward propagation for the network.
        
        Returns
        -------
        None.

        """
        L = self.num_layers
        
        # Initialize the backpropagation by computing dJ/dA of the last layer (Lth layer).
        self.compute_last_layer_dJdA()
            
        # Compute the Lth layer gradients.
        last_layer = self.layers[L]
        last_layer.layer_back_prop()
        
        # ensure dJdB is a 2D numpy array and not 1D, even though it stored as a vector, 
        # and only broadcasted into a matrix during computations.
        last_layer.gradients["dJdB"] = last_layer.gradients["dJdB"].reshape(-1,1)
        
        # Compute the gradients of the other layers.
        for l in reversed(range(1, L)):
            current_layer = self.layers[l]
            
            current_layer.layer_back_prop()
                    
            # ensure dJdB is a 2D numpy array and not 1D.
            current_layer.gradients["dJdB"] = current_layer.gradients["dJdB"].reshape(-1,1)  
    
        return None
    
    
    def update_parameters_gradient_descent(self, learning_rate):
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
    
    def train(self, num_iterations=10, batch_size=10, learning_rate=0.0000009, print_costs=True):
        
        print("Training Begins...")
        

        num_iterations=num_iterations+self.num_latest_iteration
        
        for self.num_latest_iteration in range(self.num_latest_iteration+1, num_iterations+1):
            # loop header: allows training to be resumable for the network from the state it was in 
            #  at end of its last training.
                
            # select batch from the training dataset.
            
            random_indices = np.random.choice(self.Y.shape[1], (batch_size,), replace=False)
            
            self.Y_batch=self.Y[:,random_indices]
            self.input_layer.A = self.X[:,random_indices]
                  
            # Forward propagation.
            self.network_forward_prop()
        
            # Back propagation.
            self.network_back_prop()
         
            # Update parameters.
            self.update_parameters_gradient_descent(learning_rate=learning_rate)
            
            # Compute the training cost, accuracy and precision (using the current training batch).
            
            self.training_archiver.compute_and_archive_cost(cost_type="training")
            self.training_archiver.compute_and_archive_accuracy(acc_type="training")
            self.training_archiver.compute_and_archive_precision(precis_type="training")
            
            # Compute the validation cost, accuracy and precision (using the entire training dataset).
            
            self.input_layer.A = self.X   
            self.Y_batch=self.Y
            
            self.network_forward_prop()
            self.training_archiver.compute_and_archive_cost(cost_type="validation")
            self.training_archiver.compute_and_archive_accuracy(acc_type="validation")
            self.training_archiver.compute_and_archive_precision(precis_type="validation")
            
            # rest of caching occurs here.
            
            self.training_archiver.archive_gradients()
            self.training_archiver.archive_parameters()
            
        print("Training Complete!")
            
    def evaluate(self, X, Y, metric="accuracy"):
        # assumes binary classification.
        
        _available_perfomance_metrics=["accuracy","precision"]
        
        metric=metric.lower()
        
        if not any(m == metric.lower() for m in _available_perfomance_metrics):
            raise ValueError
                
        self.input_layer.A = X
        self.Y_batch=Y
        
        self.network_forward_prop()
                
        if metric=="accuracy":
            self._compute_accuracy()
            score=self.latest_accuracy
        if metric =="precision":
            self._compute_precision()
            score=self.latest_precision
            
        return score
    
    def _compute_accuracy(self):
        Y_true=self.Y_batch.reshape(-1,)
        Y_pred=self.Y_pred.reshape(-1,)
        
        # assumes binary classification.
        self.latest_accuracy=np.average(np.where(Y_true==Y_pred, 1,0))
        
        return None
    
    def _compute_precision(self):
        Y_true=self.Y_batch.reshape(-1,)
        Y_pred=self.Y_pred.reshape(-1,)
        
        # assumes binary classification.
        mask_pred_positives = (Y_pred==1)
        self.latest_precision=np.average(np.where(Y_pred[mask_pred_positives]==Y_true[mask_pred_positives], 1, 0))
        
        return None
    
class ActivationFunction:
    
    _available_activation_funcs=["logistic","relu","tanh"]
    
    def __init__(self, name):      
        if any(a == name.lower() for a in self.__class__._available_activation_funcs):
            self.name=name.lower()
        else:
            raise ValueError
        
    def forward_pass(self, Z):
        if self.name=="logistic":
            A = self._logistic(Z)
        if self.name=="relu":
            A = self._relu(Z)
        if self.name=="tanh":
            A = self._tanh(Z) 
        return A
    
    def backward_pass(self, A,Z):
        if self.name=="logistic":
            dAdZ=self._logistic_gradient(A)
        if self.name=="relu":
            dAdZ=self._relu_gradient(Z)
        if self.name=="tanh":
            dAdZ=self._tanh_gradient(Z)          
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
                
    def set_archiving_frequencies(self, **kwargs):
        
        self.archiving_frequencies["activation"]=kwargs.get("activation", 0)
        self.archiving_frequencies["preactivation"]=kwargs.get("preactivation", 0)
        self.archiving_frequencies["cost"]=kwargs.get("cost", 0)
        self.archiving_frequencies["gradient"]=kwargs.get("gradient", 0)
        self.archiving_frequencies["parameters"]=kwargs.get("parameters", 0)
        self.archiving_frequencies["accuracy"]=kwargs.get("accuracy", 0)
        self.archiving_frequencies["precision"]=kwargs.get("precision", 0)
    
    def set_archiving_verbosities(self, **kwargs):
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
        
    def set_target_network(self, target_network):
        self.target_network=target_network
    
    def archive_activations(self):
        i = self.target_network.num_latest_iteration
        
        if i % self.archiving_frequencies["activation"] == 0:
            L = self.target_network.num_layers
            acts_all_layers=dict()
            for l in range(1, L+1):
                acts_all_layers[l]=copy.deepcopy(self.target_network.layers[l].A)
            self.all_activations[i]=acts_all_layers
    
    def archive_preactivations(self):
        i = self.target_network.num_latest_iteration
        
        if i % self.archiving_frequencies["preactivation"] == 0:
            L = self.target_network.num_layers
            preacts_all_layers=dict()
            for l in range(1, L+1):
                preacts_all_layers[l]=copy.deepcopy(self.target_network.layers[l].Z)
            self.all_preactivations[i]=preacts_all_layers        
        
    def archive_gradients(self):
        i = self.target_network.num_latest_iteration
        
        if i % self.archiving_frequencies["gradient"] == 0:
            L = self.target_network.num_layers
            grads_all_layers=dict()
            for l in range(1, L+1):
                grads_all_layers[l]=copy.deepcopy(self.target_network.layers[l].gradients)
            self.all_gradients[i]=grads_all_layers
    
    def archive_parameters(self):
        i = self.target_network.num_latest_iteration
        
        if i % self.archiving_frequencies["parameters"] == 0:
            L = self.target_network.num_layers
            params_all_layers=dict()
            for l in range (1, L+1):
                params_all_layers["W"+str(l)]=copy.deepcopy(self.target_network.layers[l].W)
                params_all_layers["B"+str(l)]=copy.deepcopy(self.target_network.layers[l].B)
            self.all_parameters[i]=params_all_layers
            
    def compute_and_archive_accuracy(self, acc_type):
        i = self.target_network.num_latest_iteration
        if i % self.archiving_frequencies["accuracy"] == 0:
            if acc_type=="training":
                self.target_network._compute_accuracy()
                self.all_training_accuracies[i]=self.target_network.latest_accuracy
                self._display_latest_if_valid(archival_target="accuracy", prefix="training")
            if acc_type=="validation":
                self.target_network._compute_accuracy()
                self.all_validation_accuracies[i]=self.target_network.latest_accuracy
                self._display_latest_if_valid(archival_target="accuracy", prefix="validation")
                
    def compute_and_archive_precision(self, precis_type):
        i = self.target_network.num_latest_iteration
        if i % self.archiving_frequencies["precision"] == 0:
            if precis_type=="training":
                self.target_network._compute_precision()
                self.all_training_precisions[i]=self.target_network.latest_precision
                self._display_latest_if_valid(archival_target="precision", prefix="training")
            if precis_type=="validation":
                self.target_network._compute_precision()
                self.all_validation_precisions[i]=self.target_network.latest_precision
                self._display_latest_if_valid(archival_target="precision", prefix="validation")
    def compute_and_archive_cost(self, cost_type):
        i = self.target_network.num_latest_iteration
        
        if i % self.archiving_frequencies["cost"] == 0:
            
            if cost_type=="training":
                self.target_network.compute_cost()
                self.all_training_costs[i] = self.target_network.cost
                self._display_latest_if_valid(archival_target="cost", prefix="training")
                
            if cost_type=="validation":
                self.target_network.compute_cost()
                self.all_validation_costs[i] = self.target_network.cost
                
                self._display_latest_if_valid(archival_target="cost", prefix="validation")

    def _display_latest_if_valid(self, archival_target, prefix=None, suffix=None):
        if prefix==None: prefix=""
        if suffix==None: suffix=""
        
        i = self.target_network.num_latest_iteration
        
        if self.archiving_verbosities[archival_target]:
            
            if archival_target=="accuracy" and prefix=="validation":
                print(prefix+" "+archival_target+", pass "+str(i)+": ",
                      helper_funcs.sigfig(self.all_validation_accuracies[i]))
            elif archival_target=="accuracy" and prefix=="training":
                print(prefix+" "+archival_target+", pass "+str(i)+": ",
                      helper_funcs.sigfig(self.all_training_accuracies[i]))                
            
            if archival_target=="cost" and prefix=="validation":
                print(prefix+" "+archival_target+", pass "+str(i)+": ",
                      helper_funcs.sigfig(self.all_validation_costs[i]))
            elif archival_target=="cost" and prefix=="training":
                print(prefix+" "+archival_target+", pass "+str(i)+": ",
                      helper_funcs.sigfig(self.all_training_costs[i]))

            if archival_target=="precision" and prefix=="validation":
                print(prefix+" "+archival_target+", pass "+str(i)+": ",
                      helper_funcs.sigfig(self.all_validation_precisions[i]))
            elif archival_target=="precision" and prefix=="training":
                print(prefix+" "+archival_target+", pass "+str(i)+": ",
                      helper_funcs.sigfig(self.all_training_precisions[i]))                