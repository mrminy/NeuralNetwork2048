import numpy as np
import theano.tensor as T
import theano.tensor.nnet as Tann
from theano import *


class HiddenLayer(object):
    """
    A fully connected hidden layer with n_in input weights and n_out output weights.
    :parameter activation is the activation function for this layer
    :parameter rand_limit_min is the minimum limit for random initialization of weights
    :parameter rand_limit_max is the maximum limit for random initialization of weights
    """
    def __init__(self, input, n_in, n_out, activation=Tann.sigmoid, rand_limit_min=-.1, rand_limit_max=.1):
        self.input = input
        self.W = theano.shared(np.random.uniform(-.1, .1, size=(n_in, n_out)))
        self.b = theano.shared(np.random.uniform(-.1, .1, size=n_out))

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))

class neuralnetwork():
    """
    Builds a neural network with topology from the layer_sizes.
    :parameter lr is the learning rate
    :parameter activation is the activation function for the network
    :parameter max_iterations is the number of training iterations trough the training set
    :parameter rand_limit_min is the minimum limit for random initialization of weights for all layers
    :parameter rand_limit_max is the maximum limit for random initialization of weights for all layers
    """
    def __init__(self, layer_sizes=[784, 24, 10], lr=.1, activation=Tann.sigmoid, max_iterations=5, rand_limit_min=-.1, rand_limit_max=.1,
                 learningSet = [], learningSet_answ = [], testSet = [], testSet_answ = []):
        print("Building network...")
        # Initializing
        # self.images, self.labels = mnist.load_all_flat_cases()  # Learning set
        # self.testing_set_images, self.testing_set_labels = mnist.load_all_flat_cases(type="testing") # Testing set

        self.learningSet = np.array(learningSet)
        self.learningSetAnsw = np.array(learningSet_answ)
        self.testSet = np.array(testSet)
        self.testSetAnsw = np.array(testSet_answ)

        self.lrate = lr  # learning rate
        self.layers = []
        self.build_ann(layer_sizes, activation=activation, rand_limit_min=rand_limit_min, rand_limit_max=rand_limit_max)  # building the neural net

        # Training the net
        print("Training network...")
        self.train_epochs(outputnodes=layer_sizes[-1], max_iterations=max_iterations, inputs=self.learningSet,
                          answers=self.learningSetAnsw, test_set=self.testSet, test_set_answers=self.testSetAnsw)

    def build_ann(self, layer_sizes=[784, 24, 10], activation=Tann.sigmoid, rand_limit_min=-.1, rand_limit_max=.1):
        """
        Builds a neural network with topology from the layer_sizes.
        :parameter activation is the activation function for the network
        :parameter rand_limit_min is the minimum limit for random initialization of weights for all layers
        :parameter rand_limit_max is the maximum limit for random initialization of weights for all layers
        """
        params = []
        inputs, answers = T.dmatrices('input', 'answers')
        assert len(layer_sizes) >= 2
        for i in range(len(layer_sizes) - 1):
            layer = HiddenLayer(inputs, layer_sizes[i], layer_sizes[i + 1], activation=activation, rand_limit_min=rand_limit_min, rand_limit_max=rand_limit_max)
            # outputs.append(layer.output)
            params.append(layer.W)
            params.append(layer.b)
            self.layers.append(layer)
        previous_out = self.layers[0].output
        x_h_out = self.layers[0].output
        for i in range(len(self.layers)-1):
            layer = self.layers[i+1]
            x_h_out = Tann.sigmoid(T.dot(previous_out, layer.W) + layer.b)
            previous_out = x_h_out
        error = T.sum((answers - x_h_out) ** 2)
        gradients = T.grad(error, params)
        backprop_acts = [(p, p - self.lrate * g) for p, g in zip(params, gradients)]
        self.predictor = theano.function([inputs], [x_h_out])
        self.trainer = theano.function([inputs, answers], error, updates=backprop_acts)

    def train_epochs(self, outputnodes, max_error=0.05, max_iterations=10, inputs=[], answers=[], test_set=[], test_set_answers=[]):
        """
        Trains the network for number of epochs of the training set
        :parameter inputs is the training set inputs
        :parameter answers is the training set answers
        :parameter test_set is the test set inputs
        :parameter test_set_answers is the test set answers
        """
        inputs = np.array(inputs)
        answers = np.array(answers)
        test_set = np.array(test_set)
        test_set_answers = np.array(test_set_answers)

        assert len(inputs) == len(answers)
        assert len(test_set) == len(test_set_answers)

        iteration_counter = 0
        test_set_blind = self.blind_test(test_set)
        error = self.calculate_error_guessing(test_set_blind, test_set_answers)
        print("Errors:")
        print(error)
        while (error > max_error) and (iteration_counter < max_iterations):
            for i in range(len(inputs)):
                self.trainer([inputs[i]], [answers[i]])
            test_set_blind = self.blind_test(test_set)
            error = self.calculate_error_guessing(test_set_blind, test_set_answers)
            print(error)
            iteration_counter += 1

    def get_ann_prediction(self, arr):
        """
        Activate the ann with image as input
        :param image: input image
        :return: a single digit answer
        """
        result_array = self.predictor([arr])
        max_value = 0.0
        max_index = 0
        for i in range(len(result_array[0][0])):
            result = result_array[0][0][i]
            if result > max_value:
                max_value = result
                max_index = i
        return max_index

    def blind_test(self, feature_sets):
        """
        Give an array of images, get answers
        :param feature_sets: two dimensional array with test data
        :return: flat array of answers
        """
        converted_sets = np.array(feature_sets)
        out_labels = []
        for image in converted_sets:
            out_labels.append(self.get_ann_prediction(image))
        return out_labels

    def predictMovePriority(self, board_state):
        """
        :param board_state: takes one or more board states
        :return: the ANN prediction of the best moves
        """
        converted_sets = np.array(board_state)
        for board in converted_sets:
            return self.predictor([board])[0]


    def calculate_error_guessing(self, guesses, labels):
        """
        Calculates the percent error from several guesses
        """
        assert len(guesses) == len(labels)
        error_counter = 0
        for i in range(len(guesses)):
            if guesses[i] != labels[i] : error_counter += 1
        return error_counter / float(len(guesses))

def relu(x):
    """
    Activation function
    """
    return theano.tensor.switch(x<0, 0, x) # possible: x * (x > 0)