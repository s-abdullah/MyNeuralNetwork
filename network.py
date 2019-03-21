import numpy as np

INIT = 0.01
RELU = 0.0000

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

  # linear computation
  # A = WX + B
  def affineForward(self, W, A, bias):
    return np.add(np.dot(W, A), bias)

  # nonlinear computation
  # Z = ReLU(A)
  def activationForward(self, A):
    return np.maximum(A, RELU)

  def jacobian_relu(self, matrix):
    return 1. * (matrix > RELU)

  def softmax(self, Z):
    r, c = Z.shape
    raisedE = np.exp(Z)
    summed = np.sum(raisedE, axis=0)
    for x in range(c):
      raisedE[:, x] /= summed[x]
    return raisedE

  # my versions very not as clean so ended up using this
  #cost function deepnotes.io/softmax-crossentropy
  def loss(self, X, y):
    m = y.shape[0]
    p = self.softmax(X)
    p = p.T
    log_likelihood = -np.log(p[range(m), y])
    loss = (np.sum(log_likelihood)) / m
    return loss

  #derivative of cost wrt softmax deepnotes.io/softmax-crossentropy
  def delta_cross_entropy(self, X, y):
      m = y.shape[0]
      grad = self.softmax(X)
      grad = grad.T
      grad[range(m), y] -= 1
      grad = grad/m
      return grad.T


  # helper class functions
  def clean(self):
    del self.dWeights[:]
    del self.dBias[:]
    del self.linearA[:]
    del self.nonlinearZ[:]
    del self.output[:]

  def printWB(self):
    for thing in self.weights:
      print(thing.shape, thing)

    print("bias Matrices: ")
    for thing in self.bias:
      print(thing.shape, thing)

  def printInter(self):
    print("Dweights Matrices: ")
    for thing in self.dWeights:
      print(thing.shape, thing)

    print("Dbias Matrices: ")
    for thing in self.dBias:
      print(thing.shape, thing)

    print("A Matrices: ")
    for thing in self.linearA:
      print(thing.shape, thing)

    print("Z Matrices: ")
    for thing in self.nonlinearZ:
      print(thing.shape, thing)

    print("output Matrices: ")
    for thing in self.output:
      print(thing.shape, thing)
