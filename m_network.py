# Fully matrix based neural network

import numpy as np
import time

def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))
def cost_activation_derivative(activation, answer):
    return (activation - answer)
def vectorise_answer(answer):
    vector = np.zeros((10,1))
    vector[answer] = 1.0
    return vector

class Network:

    def __init__(self, layers):

        self.layers_list = layers
        self.layers = len(layers)
        self.weights = [np.random.randn(to_size, from_size) for from_size,to_size in zip(layers[:-1],layers[1:])]
        self.biases = [np.random.randn(layer_size,1) for layer_size in layers[1:]]

    def stochasticGD(self,batch_sizes,learning_rate,epochs,character_training_data,evaluation_data):
        data_amount = len(character_training_data)


        for epoch in range(epochs):
            start_time = time.time()
            
            np.random.shuffle(character_training_data)
            batches = np.array_split(character_training_data,int(data_amount/batch_sizes))
            self.update_batches(batches,learning_rate,batch_sizes)

            evaluation = self.check(evaluation_data)
            print("Epoch [{}]: {}/{} , took {:.0f}ms".format(epoch+1,
                                                             evaluation,
                                                             len(evaluation_data),
                                                             ((time.time()-start_time)*1000)))

        
    def update_batches(self,batches,learning_rate,batch_sizes):
        
        for batch in batches:

            inputs = np.array([inputs.flatten() for inputs, answer in batch]).T
            answers = np.array([vectorise_answer(answer).flatten() for inputs, answer in batch]).T
                
            weights_change, biase_change = self.backpropagate(inputs,answers)

            self.weights = [(weight - (learning_rate/len(batch)) * change)
                            for weight,change in zip(self.weights,weights_change)]
            self.biases = [(biases - (learning_rate/len(batch)) * change)
                           for biases,change in zip(self.biases, biase_change)]

    def backpropagate(self, inputs, answer):
        
        activations = [inputs]
        weighted_inputs = []

        new_activation = inputs
        
        for index in range(self.layers - 1):
            weighted_input = np.array(np.dot(self.weights[index],new_activation) + self.biases[index])
            
            weighted_inputs.append(weighted_input)
            new_activation = np.array(sigmoid(weighted_input))
            activations.append(new_activation)

        output_error = np.multiply(cost_activation_derivative(activations[-1],answer),
                                   sigmoid_derivative(weighted_inputs[-1]))
        
        weights = list(np.copy(self.weights))
        biases = list(np.copy(self.biases))

        weights[-1] = np.dot(output_error,activations[-2].T)
        biases[-1] = np.array([np.sum(output_error,1)]).T
        
        for layer in range(self.layers - 2, 0, -1):
            output_error = np.multiply(np.dot(self.weights[layer].T,output_error),
                                       sigmoid_derivative(weighted_inputs[layer - 1]))

            weights[layer - 1] = np.dot(output_error,activations[layer - 1].T)
            biases[layer - 1] = np.array([np.sum(output_error,1)]).T
            
        return weights, biases

    def check(self,evaluation_data):

        total = 0
        for inputs, answer in evaluation_data:
            output = inputs
            for index in range(self.layers - 1):
                weighted_input = np.array(np.dot(self.weights[index],output) + self.biases[index])
                output = np.array(sigmoid(weighted_input))
            if np.argmax(output) == answer:
                total += 1
        return total
