"""
@author: Prince Okoli

"""
import numpy as np
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
        
        self.A=None
        self.Z=None
        self.W=None
        self.B=None
        
        self.cost=None
        
        self.dAdZ=None
        self.dJdA=None
        self.dJdZ=None
        self.dJdW=None
        self.dJdB=None
        
    def _logistic(self, Z):
        A = 1/(1+np.exp(-Z))
        return A
    
    def _logistic_gradient(self, A):
        dAdZ = A * (1-A)
        return dAdZ
    
    def _forward(self):
        self.Z = np.matmul(self.W, self.X_batch) + self.B
        self.A=self._logistic(self.Z)
    
    def _backward(self):
        m = self.X.shape[1]
        self.dAdZ=self._logistic_gradient(self.A)
        self.dJdA = - ((self.Y_batch / self.A) - ((1 - self.Y_batch) / (1 - self.A)))
        
        self.dJdZ = self.dAdZ * self.dJdA
        
        self.dJdW=(1/m) * np.matmul(self.dJdZ, self.X_batch.T) 
        self.dJdB= np.sum(self.dJdZ, axis=1)
        
    def _update_parameters_via_gradient_descent (self, learning_rate):
        self.W = self.W - learning_rate * self.dJdW
        self.B = self.B - learning_rate * self.dJdB
        
    def _initialize_parameters(self, random_seed=11):
        prng=np.random.RandomState(seed=random_seed)
        n=self.X.shape[0]
        self.W=prng.random(size=(1, n))
        self.B=prng.random(size=(1, 1))
        
    def _compute_accuracy(self):
        Y_pred=np.where(self.A>0.5, 1, 0)
        Y_true=self.Y_batch
        
        accuracy=np.average(np.where(Y_true==Y_pred, 1, 0))
        
        return accuracy
    
    def _compute_precision(self):
        Y_true=self.Y_batch
        Y_pred=np.where(self.A>0.5, 1, 0)
        
        pred_positives_mask = (Y_pred==1)
        precision=np.average(np.where(Y_pred[pred_positives_mask]==Y_true[pred_positives_mask]))        
        
        return precision
    
    def train(self,num_iterations, learning_rate, batch_size):
        print("Training begins...")
        self._initialize_parameters()
        for i in range(0, num_iterations):
            random_indices = np.random.choice(self.Y.shape[1], (batch_size,), replace=False)
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