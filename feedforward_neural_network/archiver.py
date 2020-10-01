# -*- coding: utf-8 -*-
"""
@author: Prince Okoli

Controls for the components of the network and performance metric to cache and 
display during training.

"""
import numpy as np
import copy
np.seterr(over="warn", under="warn") # warn for overflows and underflows.

# import custom modules.
from . import _helper_funcs

# Check whether major version has changed, 
# and print warning to console if necessary.
_helper_funcs.check_numpy_ver()

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
        
    def set_target_network(self, target_network): # module-private.
        self.target_network=target_network
    
    def archive_activations(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["activation"]!=0) and (i % self.archiving_frequencies["activation"] == 0):
            L = self.target_network.num_layers
            acts_all_layers=dict()
            for l in range(1, L+1):
                acts_all_layers[l]=copy.deepcopy(self.target_network.layers[l].A)
            self.all_activations[i]=acts_all_layers
    
    def archive_preactivations(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["preactivation"]!=0) and (i % self.archiving_frequencies["preactivation"] == 0):
            L = self.target_network.num_layers
            preacts_all_layers=dict()
            for l in range(1, L+1):
                preacts_all_layers[l]=copy.deepcopy(self.target_network.layers[l].Z)
            self.all_preactivations[i]=preacts_all_layers        
        
    def archive_gradients(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["gradient"]!=0) and (i % self.archiving_frequencies["gradient"] == 0):
            L = self.target_network.num_layers
            grads_all_layers=dict()
            for l in range(1, L+1):
                grads_all_layers[l]=copy.deepcopy(self.target_network.layers[l].gradients)
            self.all_gradients[i]=grads_all_layers
    
    def archive_parameters(self): # module-private.
        i = self.target_network.num_latest_iteration
        
        if (self.archiving_frequencies["parameters"]!=0) and (i % self.archiving_frequencies["parameters"] == 0):
            L = self.target_network.num_layers
            params_all_layers=dict()
            for l in range (1, L+1):
                params_all_layers["W"+str(l)]=copy.deepcopy(self.target_network.layers[l].W)
                params_all_layers["B"+str(l)]=copy.deepcopy(self.target_network.layers[l].B)
            self.all_parameters[i]=params_all_layers
            
    def compute_and_archive_accuracy(self, acc_type):
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
                
    def compute_and_archive_precision(self, precis_type): # module-private.
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
                
    def compute_and_archive_cost(self, cost_type):
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
                      str(_helper_funcs.sigfig(self.all_validation_accuracies[i]))+"\n"
            elif archival_target=="accuracy" and prefix=="training":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(_helper_funcs.sigfig(self.all_training_accuracies[i]))+"\n"                
            
            if archival_target=="cost" and prefix=="validation":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(_helper_funcs.sigfig(self.all_validation_costs[i]))+"\n"
            elif archival_target=="cost" and prefix=="training":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(_helper_funcs.sigfig(self.all_training_costs[i]))+"\n"

            if archival_target=="precision" and prefix=="validation":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(_helper_funcs.sigfig(self.all_validation_precisions[i]))+"\n"
            elif archival_target=="precision" and prefix=="training":
                self.report += prefix+" "+archival_target+", iter. "+str(i)+": "+\
                      str(_helper_funcs.sigfig(self.all_training_precisions[i]))+"\n"
    def clear_report(self):
        self.report=""
    def print_report(self):
        print(self.report,"="*10)