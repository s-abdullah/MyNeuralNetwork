import numpy as np
# only for downloading data
import torch, torchvision

import matplotlib.pyplot as plt
import pickle
from network import *
from utils import *


# global varibales
BATCHSIZE = 50
LEARNINGRATE = 0.03
OUTPUTS = 10
IMAGESIZE = 3072
LAYERS = [300, 250]
ITERATIONS = 20
INIT = 0.01
RELU = 0.0000




# main function used to call the network
def main():
  # downloading the dataset
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(root="/data", train=True,download=True, transform=transform)

  testset = torchvision.datasets.CIFAR10(root="/data", train=False,download=True, transform=transform)

  train_set, train_set_label, validation_set, validation_set_label = train_val_split(trainset)
  flat_test, labels = process_test_set(testset)


  print("The splits are: ")
  print(train_set.shape, train_set_label.shape)
  print(validation_set.shape)

  # myNN = NeuralNetwork(OUTPUTS, IMAGESIZE, BATCHSIZE, LEARNINGRATE, LAYERS)
  
  # # starting the training
  # myNN.train(train_set, train_set_label, validation_set, validation_set_label, ITERATIONS)
  # myNN.save()
  # myNN.clean()





if __name__ == "__main__":
	main()