import numpy as np

# function flattens the input array
def flatten(thing):
  return thing.numpy().flatten()



def train_val_split(trainset):
  
  #create a numpy array with all the data (each entry array[i] is equal to trainset[i][0])
  data = []
  labels = []
  for i in range(50000):
    data.append(trainset[i][0].numpy())
    labels.append(trainset[i][1])

  
  data = np.array(data)
  labels = np.array(labels)
  
  # getting curent state so shuffle is the same for both arrays
  curState = np.random.get_state()
  np.random.shuffle(data)
  # setting the state
  np.random.set_state(curState)
  np.random.shuffle(labels)
  
  #images are unflattened
  temp_val = data[:5000]
  val_label = labels[:5000]
  temp_train = data[5000:]
  train_label = labels[5000:]
  
  #flatten the images
  val = []
  train = []
  
  for element in temp_val:
    val.append(element.flatten())
  
  for element in temp_train:
    train.append(element.flatten())
  
  train = np.array(train)
  val = np.array(val)
  
  return train.T, train_label.T, val.T, val_label.T


def process_test_set(testset):

  #create a numpy array with all the data (each entry array[i] is equal to trainset[i][0])
  data = []
  labels = []

  for i in range(len(testset)):
    data.append(testset[i][0].numpy())
    labels.append(testset[i][1])

  data = np.array(data)
  labels = np.array(labels)
  #flatten the images
  flat_test = []

  for element in data:
    flat_test.append(element.flatten())

  flat_test = np.array(flat_test)

  return flat_test, labels