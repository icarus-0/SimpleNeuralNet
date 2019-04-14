import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)



trainig_input  = np.array([[0,0,1],
                        [1,1,1],
                        [1,0,1],
                        [0,1,1]])

trainig_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

print('Random starting synaptic_weights:')

synaptic_weights = 2 + np.random.random((3,1)) - 1

print('Random starting synaptic_weights:')
print(synaptic_weights)

for iteration in range(10000):

    input_layer = trainig_input

    outputs = sigmoid(np.dot(input_layer,synaptic_weights))

    error  = trainig_outputs - outputs

    adjustment = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T , adjustment)
    print('training outputs:')
    print(error)

print('synaptic_weights after traing')
print(synaptic_weights)

print('outputs after the traing')
print(outputs)
