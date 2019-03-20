import numpy as np

INIT = 0.01

class NeuralNetwork:

  # the input is the a list of the dimensions of the hidden layers, the len of the list will be the number of hidden layers
  # Additionally it will take the number of classes and the size of the flattened image as input which will be 3072 for CIFAR10
  # the hyper parameters are also passed here, training is just for training, all settings to be done in initi

  def __init__(self, numOutput, imageSize, batchSize, learningRate, layer_dimensions):

    # this will store the hidden layer weghts, initialized to random
    self.weights = []
    # derivative of the weights
    self.dWeights = []

    # this will store the bias vectors
    self.bias = []
    # derivative of the bias
    self.dBias = []

    # to store the intermediate variables in the computation
    self.linearA = []
    self.nonlinearZ = []
    self.output = []
    self.input = []

    self.batch = batchSize
    self.alpha = learningRate
    self.numLayers = len(layer_dimensions)

    prevLayer = imageSize
    # irterating over the hidden layers sizes
    for theSize in layer_dimensions:
      self.weights.append(
          (np.random.rand(theSize, prevLayer)*INIT).astype(np.float128))
      self.bias.append(
          (np.random.rand(theSize, batchSize)*INIT).astype(np.float128))
      prevLayer = theSize

    # adding weights for the output layer
    self.weights.append(
        (np.random.rand(numOutput, prevLayer)*INIT).astype(np.float128))
    self.bias.append(
        (np.random.rand(numOutput, batchSize)*INIT).astype(np.float128))

    print("Neural Network is intialized with the follwing setting: ")
    print("Learning rate: ", self.alpha)
    print("Batchsize: ", self.batch)
    print("Number of Layers: ", self.numLayers)
    print("Weight Matrices: ")
