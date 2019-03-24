import numpy as np
# only for downloading data
import torch, torchvision

import pickle
from network import *
import utils as ut


# global varibales
BATCHSIZE = 50
LEARNINGRATE = 0.03
OUTPUTS = 10
IMAGESIZE = 3072
LAYERS = [300, 250]
ITERATIONS = 20
wFile = "w"
bFile = "b"




# main function used to call the network
def main():
  # downloading the dataset
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(root="/data", train=True,download=True, transform=transform)

  testset = torchvision.datasets.CIFAR10(root="/data", train=False,download=True, transform=transform)

  train_set, train_set_label, validation_set, validation_set_label = ut.train_val_split(
      trainset)
  flat_test, labels = ut.process_test_set(testset)


  print("The splits are: ")
  print(train_set.shape, train_set_label.shape)
  print(validation_set.shape)

  #training
  myNN = NeuralNetwork(OUTPUTS, IMAGESIZE, BATCHSIZE, LEARNINGRATE, LAYERS)
  
  # # starting the training
  myNN.train(train_set, train_set_label, validation_set, validation_set_label, ITERATIONS)
  myNN.save()
  myNN.clean()

  # testing
  myNNtest = NeuralNetwork(OUTPUTS, IMAGESIZE, BATCHSIZE, LEARNINGRATE, LAYERS)
  myNNtest.load(wFile, bFile)
  print("validation accuracy for current", myNNtest.check(
    validation_set, validation_set_label))
  print("train accuracy for current", myNNtest.check(train_set, train_set_label))

  images, label, predictions = myNNtest.pred(flat_test.T, labels)


  classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
  for x in range(10):
    ut.imshow(testset[x][0])
    print("Ground Truth: ", label[x], classes[label[x]])
    print("prediction: ", predictions[x], classes[predictions[x]])



if __name__ == "__main__":
	main()
