import artificial_neuron as an
import numpy as np

data_train=np.genfromtxt("../datasets/toy_dataset1/toy_dataset_velocity_ke_train.csv", delimiter=",").T

data_test=np.genfromtxt("../datasets/toy_dataset1/toy_dataset_velocity_ke_test.csv", delimiter=",").T

x_train=data_train[0:2, :]
y_train=data_train[2, :].reshape(1,-1)

x_test=data_test[0:2, :]
y_test=data_test[2, :].reshape(1,-1)

neuron=an.Neuron(X=x_train, Y=y_train)
neuron.train(num_iterations=50, learning_rate=0.007, batch_size=1024)

print(neuron.evaluate(X=x_train, Y=y_train))
print(neuron.evaluate(X=x_test, Y=y_test))

#%%