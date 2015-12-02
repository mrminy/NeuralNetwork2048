import random
import numpy as np
import theano.tensor.nnet as Tann
import theano.tensor as T
import ANN
import TrainingSetReader as reader


class ANNPlayer:
    """
    Defines a player who makes move by training an Artificial Neural Network
    """
    def __init__(self, layer_sizes=[16, 2000, 4], lr=.1, activation=ANN.relu, max_iterations=10, rand_limit_min=-.02, rand_limit_max=.02,
                 learningSet = [], learningSet_answ = [], testSet = [], testSet_answ = []):

        # Converting data...
        print("Converting data...")
        learningSet = self.convert_input_divide_relative_to_max(learningSet)
        learningSet_answ = self.convert_answers(learningSet_answ)
        testSet = self.convert_input_divide_relative_to_max(testSet)

        # Builds the network
        self.neuralNet = ANN.neuralnetwork(layer_sizes=layer_sizes, lr=lr, activation=activation, max_iterations=max_iterations, rand_limit_min=rand_limit_min,
                                           rand_limit_max=rand_limit_max, learningSet=learningSet, learningSet_answ=learningSet_answ, testSet=testSet, testSet_answ=testSet_answ)

    def convert_input_divide_relative_to_max(self, input):
        """
        Devides the input by the max value in the input
        :param input: array of 16 values (exponents of 2)
        :return: converted array
        """
        assert len(input[0]) == 16
        arr = []
        for i in range(len(input)):
            innerArr = input[i]
            innerArr = np.array(innerArr)
            maxValue = innerArr.max()
            innerArr /= maxValue
            arr.append(innerArr)
        return np.array(arr)

    def convert_input_divide_by_2048(self, input):
        """
        Devides the input by 11 (2^11 = 2048)
        :param input: array of 16 values (exponents of 2)
        :return: converted array
        """
        assert len(input[0]) == 16
        arr = []
        maxValue = 11.0
        for i in range(len(input)):
            innerArr = input[i]
            innerArr = np.array(innerArr)
            innerArr /= maxValue
            arr.append(innerArr)
        return np.array(arr)

    def convert_input_divide_relative_to_max_pow_2(self, input):
        """
        Makes power of 2's and then devides the input by the max value
        :param input: array of 16 values (exponents of 2)
        :return: converted array
        """
        assert len(input[0]) == 16
        arr = []
        for i in range(len(input)):
            innerArr = input[i]
            innerArr = np.array(innerArr)
            for j in range(len(innerArr)):
                if not innerArr[j]==0:
                    innerArr[j] = pow(2, innerArr[j])
            maxValue = innerArr.max()
            innerArr /= maxValue
            arr.append(innerArr)
        return np.array(arr)


    def convert_answers(self, answers):
        """
        Converts the answer to an array of 0 and 1
        :param array of integers
        :return double array of arrays containing 0 and 1
        """
        arr = []
        for i in range(len(answers)):
            innerArr = np.zeros(4)
            innerArr[answers[i]] = 1.0
            arr.append(innerArr)
        return arr

    def getMove(self, boardValues):
        """
        Converts the predicted array and returns the indexes in order [0.1, 0.6, 0.05, 0.25] --> [2, 4, 1, 3]
        :param boardValues: a board state
        :return: an array of moves in prioritized order
        """
        arr = self.convert_input_divide_relative_to_max([boardValues])
        result = self.neuralNet.predictMovePriority(arr)[0]
        movePriority = []
        for k in range(len(result)):
            max_value = -1.0
            max_index = -1.0
            for i in range(len(result)):
                if result[i] > max_value:
                    max_index = i
                    max_value = result[i]
            result[max_index] = -1
            movePriority.append(max_index+1)
        return movePriority

class RandomPlayer:
    """
    A random player
    """
    def getMove(self, boardValues):
        """
        :param boardValues: a board state
        :return: a random move priority
        """
        movePriority = []
        while len(movePriority) < 4:
            value = random.choice([1,2,3,4])
            if value not in movePriority:
                movePriority.append(value)
        return movePriority


class Player:
    """
    Common class for a player
    Could be a random player of an ANN player.
    Can be expanded to human player as well
    """
    def __init__(self, randomPlayer = False, layer_sizes=[16, 250, 4], lr=.1, activation=ANN.relu, max_iterations=1000,
                 rand_limit_min=-.02, rand_limit_max=.02, learning_set="humantest2", test_set="humantest1"):
        if randomPlayer:
            self.player = RandomPlayer()
        else:
            print("Reading data...")
            trainingSet, trainingSetAnsw = reader.readSet(filename=learning_set)
            testSet, testSetAnsw = reader.readSet(filename=test_set)
            self.player = ANNPlayer(layer_sizes=layer_sizes, lr=lr, activation=activation, max_iterations=max_iterations,
                                    rand_limit_min=rand_limit_min, rand_limit_max=rand_limit_max, learningSet=trainingSet,
                                    learningSet_answ=trainingSetAnsw, testSet=testSet, testSet_answ=testSetAnsw)

    def move(self, boardValues):
        """
        Common move method
        :param boardValues: a board state
        :return: a prioritized move array
        """
        return self.player.getMove(boardValues=boardValues)