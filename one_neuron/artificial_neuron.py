"""
@author: Prince Okoli

"""
import numpy as np
np.seterr(over="warn", under="warn") # warn for overflows and underflows.

class Neuron:
    """
    Artificial nueron for machine learning.
    
    DATA MEMBERS
    ------------
    
    """
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y
        
        self.X_batch=None
        self.Y_batch=None
        
        self.a=None
        self.z=None
        self.w=None
        self.b=None
                
        self.dAdZ=None
        self.dJdA=None
        self.dJdZ=None
        self.dJdW=None
        self.dJdB=None
        
    def _logistic(self, z):
        a = 1/(1+np.exp(-z))
        return a
    
    def _logistic_gradient(self, a):
        dAdZ = a * (1-a)
        return dAdZ
        
    def _forward(self):
        self.z = np.matmul(self.w, self.X_batch) + self.b
        self.a=self._logistic(self.z)
    
    def _backward(self):
        m = self.X.shape[1]
        self.dAdZ=self._logistic_gradient(self.a)
        self.dJdA = - ((self.Y_batch / self.a) - ((1 - self.Y_batch) / (1 - self.a)))
        
        self.dJdZ = self.dAdZ * self.dJdA
        
        self.dJdW=(1/m) * np.matmul(self.dJdZ, self.X_batch.T) 
        self.dJdB= np.average(self.dJdZ, axis=1)
        
    def _update_parameters_via_gradient_descent (self, learning_rate):
        self.w = self.w - learning_rate * self.dJdW
        self.b = self.b - learning_rate * self.dJdB
        
    def _initialize_parameters(self, random_seed=11):
        prng=np.random.RandomState(seed=random_seed)
        n=self.X.shape[0]
        self.w=prng.random(size=(1, n))*0.01
        self.b=np.zeros(shape=(1, 1))
        
    def _compute_accuracy(self):
        
        if np.isnan(self.a).all():
            print("Caution: All the activations are null values.")
            return None

        Y_pred=np.where(self.a>0.5, 1, 0)
        Y_true=self.Y_batch
        
        accuracy=np.average(np.where(Y_true==Y_pred, 1, 0))
        
        return accuracy
    
    def _compute_precision(self):
        
        if np.isnan(self.a).all():
            print("Caution: All the activations are null values.")
            return None
        
        Y_true=self.Y_batch
        Y_pred=np.where(self.a>0.5, 1, 0)
        
        pred_positives_mask = (Y_pred==1)
        precision=np.average(np.where(Y_pred[pred_positives_mask]==Y_true[pred_positives_mask]))        
        
        return precision
    
    def train(self,num_iterations, learning_rate, batch_size, random_seed=11):
        print("Training begins...")
        self._initialize_parameters(random_seed=random_seed)
        prng=np.random.RandomState(seed=random_seed)
        for i in range(0, num_iterations):
            random_indices = prng.choice(self.Y.shape[1], (batch_size,), replace=False)
            self.Y_batch = self.Y[:,random_indices]
            self.X_batch = self.X[:,random_indices]
            
            self._forward()
            self._backward()
            
            self._update_parameters_via_gradient_descent(learning_rate=learning_rate)
            
        print("Training Complete!")
            
    def evaluate(self, X, Y, metric="accuracy"):
        
        _available_perfomance_metrics=["accuracy","precision"]
        
        metric=metric.lower()
        
        if not any(m == metric.lower() for m in _available_perfomance_metrics):
            raise ValueError
                
        self.X_batch = X
        self.Y_batch = Y
        
        self._forward()
                
        if metric=="accuracy":
            score=self._compute_accuracy()
        if metric =="precision":
            score=self._compute_precision()
            
        return score
