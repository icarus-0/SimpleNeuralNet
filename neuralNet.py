import numpy as np
import perceptron

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3,1)) - 1


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivatives(self,x):
        return  x *(1-x)

    def train(self,traing_inputs, traing_outputs,traing_iterations):

        for iteration in range(traing_iterations):
            output = self.think(traing_inputs)

            error = traing_outputs - output

            adjustments = np.dot(traing_inputs.T , error*self.sigmoid_derivatives(output))

            self.synaptic_weights += adjustments


    def think(self,input):

            input = input.astype(float)

            output = self.sigmoid(np.dot(input,self.synaptic_weights))

            return  output

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("random syanptic weight")
    print(neural_network.synaptic_weights)

    traing_inputs = np.array([[0, 0, 1],
                              [1, 1, 1],
                              [1, 0, 1],
                              [0, 1, 1]])

    traing_outputs = np.array([[0, 1, 1, 0]]).T

    neural_network.train(traing_inputs, traing_outputs, 10000)

    print("synaptic weigt after training")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New Situation : input data=", A, B, C)

    print("output data: ")
    print(neural_network.think(np.array([A, B, C])))



