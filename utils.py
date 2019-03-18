







# function flattens the input array
def flatten(thing):
  return thing.numpy().flatten()


"""
split the training data into train and validation. 10% i.e. 5000 images will be used for validation

call with UNFLATTENED training data i.e. trainset

returns two np arrays:

  arr1 is train with dimensions (45000,3072)
  arr2 is validation with dimensions (5000,3072)

"""
def train_val_split(trainset):
  
  #create a numpy array with all the data (each entry array[i] is equal to trainset[i][0])
  data = []
  labels = []
  for i in range(50000):
    data.append(trainset[i][0].numpy())
    labels.append(trainset[i][1])

  
  data = np.array(data)
  labels = np.array(labels)
  
#   print(data.shape) # outputs (50000, 3, 32, 32)
#   print(labels.shape) # ouputs (5000, 1)
  
  #split the data
  
  number_of_validation_points = 5000
  
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
  
  #print(len(train)) #45000
  #print(len(val)) #5000
  
  train = np.array(train)
  val = np.array(val)
  
#   print("The splits are: ")
#   print(train.shape) #(45000, 3072)   
#   print(val.shape) #(5000, 3072) 
  
  return train.T, train_label.T, val.T, val_label.T


