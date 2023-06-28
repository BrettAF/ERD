
number_of_windows = 20000
window_size = 600
percent_testing=.1
percent_validation=.1
batch_size = 128
number_of_epochs=200

import os
import pandas as pd
import numpy as np

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
import sklearn
from tensorflow.python import training

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
import random as r
import scipy as sp
from scipy.stats import zscore
from scipy.fft import fft, ifft

from keras.layers.rnn import LSTM

time_between_signals = 3.906266
#This window size generates a number of samplings in a window of specific length
window_time= (window_size*time_between_signals)/1000

validation_windows=int(number_of_windows*(percent_validation))
testing_windows=int(number_of_windows*(percent_testing))
training_windows=number_of_windows-testing_windows-validation_windows

print("This will generate:\n",training_windows , " windows for training,\n",
      validation_windows,"windows for validation &\n",
      testing_windows,"Windows for testing.\n",
      "Each window has",window_time,"seconds, and",window_size,"frames\n",
      "This is about",number_of_windows*window_time/60/60,"hours of data")

"""# Reading & Shaping Data"""

dfs=[]
for x in os.listdir():
    if x.endswith("formated.csv"):
      print(x)
      dfs.append( pd.read_csv(str(x)))

#for recording the lables of the dataset as a one-hot vector
def LabelVector3(label):
  zero_vector=[1,0,0]
  one_vector =[0,1,0]
  two_vector =[0,0,1]

  if label==0:
      return zero_vector
  elif label ==1:
      return one_vector
  else: return two_vector

def LabelVector2(label):
  zero_vector=[1,0]
  one_vector =[0,1]
  if label==0:
      return zero_vector
  else :
      return one_vector

#This ensures that no two training windows are the same.
#It generates a vector of shuffled unique start locations
def vectorMakerV2():
  arr=[]
  for df in range(0,len(dfs)):
    for row in range(0,len(dfs[df])-window_size-1,window_size):
        arr.append([df,row])
  np.random.shuffle(arr)
  arr=np.array(arr)

  window_df_starts = list(arr[:,0])
  window_starts= list(arr[:,1])
  return window_starts,window_df_starts

def frameArrayMakerV2(i, window_length,  window_starts, window_df_starts,normalize,fft_flag, include_seizures):
  frames_array=[]
  label_array=[]
  label_OHV=[]

  while i<window_length:
    fft_array=[]
    start=window_starts[i]
    df=window_df_starts[i]
    i=i+1
    #Still ordered data from a dataframe of a specific size
    new_df=(dfs[df]).loc[start:start+window_size-1]

    #Stores the lables in an array
    new_label=new_df.at[start+window_size-1,"Label"]

    if ((include_seizures == True) or (new_label != 2) ):
      new_df=new_df.drop(columns=["EEG Time", "Label"])


      # normalizes the columns
      if normalize == True:
          new_df = zscore(new_df)
      label_array.append(new_label)

      if (include_seizures == True):
          label_OHV.append( LabelVector3(new_label))
      else:
          label_OHV.append( LabelVector2(new_label))

        #if fft_flag == True:
          #for j in range(0,16):
            #one_channel=new_df[:,j]
            #anfft=fft(one_channel)
            #fft_array.append(anfft.real[20:window_size//2])
            #fft_array.append(anfft.imag[20:window_size//2])
          #new_df=fft_array
      frames_array.append( new_df)


  return frames_array, label_array, label_OHV,i

#This function guarantees that the number of each training label should stay aproximately even
def rebalanceRatio(training_ds_p, training_lables_p,training_labels_OHV_p):

  training_ds=training_ds_p[0:training_windows]
  training_lables=training_lables_p[0:training_windows]
  training_labels_OHV=training_labels_OHV_p[0:training_windows]
  ratio=training_lables.count(0)/training_windows

  a=training_windows
  b=0
  while ratio >.50:
    while training_lables_p[a] !=1:
      a+=1
    while training_lables[b] !=0:
      b+=1
    training_lables[b]=training_lables_p[a]
    training_ds[b]=training_ds_p[a]
    training_labels_OHV[b]=training_labels_OHV_p[a]
    a+=1
    b+=1
    ratio=training_lables.count(0)/training_windows
  return training_ds,training_lables,training_labels_OHV

#This generates the windows for training, testing, and validation.
#If hot vectors it true it will return your labels in the form of a a one-hot-vector
#If print ratio is true, it will print the number of each type of label and their ratio
#if normalize is true, it will perform z-score normalization
def get_dataset_partitions_tf(dfs, hot_vector=True, print_ratio = True, normalize=False, fft=False, include_seizures = True):

    window_starts,window_df_starts= vectorMakerV2()

    #For testing data
    testing_ds, testing_lables,testing_labels_OHV,i =frameArrayMakerV2(0,testing_windows+1,  window_starts,window_df_starts,normalize,fft,include_seizures)

    #For validation data
    validation_ds, validation_lables,validatiion_labels_OHV,i =frameArrayMakerV2(i,validation_windows+testing_windows+1,  window_starts,window_df_starts,normalize,fft,include_seizures)

     #for training data
    training_ds_p, training_lables_p,training_labels_OHV_p,i =frameArrayMakerV2(i,len(window_starts), window_starts,window_df_starts,normalize,fft,include_seizures)
    #print(i,window_starts[30],window_df_starts[30])
    #print(testing_ds[30].head)

    training_ds, training_lables,training_labels_OHV = rebalanceRatio(training_ds_p, training_lables_p,training_labels_OHV_p)




    if print_ratio == True:
      print("# of 0:",training_lables.count(0),",",training_lables.count(0)/training_windows)
      print("# of 1:",training_lables.count(1),",",training_lables.count(1)/training_windows)
      print("# of 2:",training_lables.count(2),",",training_lables.count(2)/training_windows)
    if hot_vector == True:
      return training_ds, testing_ds, validation_ds, training_labels_OHV, testing_labels_OHV,validatiion_labels_OHV
    return training_ds, testing_ds, validation_ds, training_lables, testing_lables, validation_lables

#This generates the windows for training, testing, and validation.
#If hot vectors it true it will return your labels in the form of a a one-hot-vector
np.set_printoptions(threshold=np.inf)
train_ds, test_ds, valid_ds, train_lables, test_lables,  valid_lables = get_dataset_partitions_tf(dfs,print_ratio=True, normalize=True, include_seizures=False, hot_vector=True)

training_data = np.array(train_ds)
testing_data = np.array(test_ds)
validation_data= np.array(valid_ds)


training_lables= np.array(train_lables)
testing_lables= np.array(test_lables)
validation_lables=np.array(valid_lables)


print("training  ",training_data.shape,",",training_lables.shape)
print("testing   ",testing_data.shape,",",testing_lables.shape)
print("validating",validation_data.shape,",",validation_lables.shape)
print()

"""# Models & Training"""

input_shape = ( window_size, 16, 1)
training_data = training_data.reshape(-1,window_size, 16, 1)

print("training  ",training_data.shape,",",training_lables.shape)
m2Dlstm = keras.Sequential([
  layers.Conv2D(filters=32, kernel_size=(5,5) ,input_shape=input_shape),
  layers.Conv2D(filters=16, kernel_size=(3,3)),
  layers.MaxPool2D( pool_size=(50, 5)),
  layers.Reshape((22,16)),
  layers.LSTM(90, activation='relu' ),
  layers.Dense(75),
  layers.Dense(40),
  layers.Dense(training_lables.shape[1])
])
print(m2Dlstm.summary())
m2Dlstm.compile(
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# normalized .5466,
m2Dlstm.fit(training_data, training_lables, batch_size=batch_size, epochs=number_of_epochs, validation_data=(validation_data, validation_lables))

"""# Evaluating

"""

test_loss, test_acc = m2Dlstm.evaluate(testing_data,testing_lables, verbose=2)
print(test_acc)