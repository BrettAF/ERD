# -*- coding: utf-8 -*-
"""Windowing with Smote

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zxqPb1k50P-SZjNNAPpqC2vYqhTyPVkI
"""

#from google.colab import drive
#drive.mount('/content/drive')
d = dir()

#You'll need to check for user-defined variables in the directory
for obj in d:
    #checking for built-in variables/functions
    if not obj.startswith('__'):
        #deleting the said obj, since a user-defined function
        del globals()[obj]
del obj

"""# Libraries and variables"""

import os
import pandas as pd
import numpy as np
import copy

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python import training

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
import random as r
import scipy as sp
from scipy.stats import zscore
from scipy.fft import fft, ifft

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from numpy import where

from keras.layers.rnn import LSTM

window_size = 600 #This is the length of each frame
number_of_channels=18 #The number of channels data is recieved from

percent_testing=.07 #The percentage of the data that will be used for testing
percent_validation=.07 #The percentage of the data that will be used for validation
percent_training=.86 #The percentage of the data that will be used for training

batch_size = 64
number_of_epochs=50


#This is the period before a seizure that will be examined.
Period_of_interest=60*60*1

folders_for_making_model = ('chb01',)
#,'chb02','chb03','chb04','chb05'

"""# Reading & Shaping Data"""

#This ensures that no two training windows are the same.
#It generates a vector of shuffled unique start locations
#It creates two vectors, one that indicates what array to draw data from,
#and one that indicates at what row in that dataset to start at
def vectorMakerV2():
  arr=[]
  for df in range(0,len(dfs)):
    for row in range(0,len(dfs[df])-window_size-1,window_size):
        arr.append([df,row])
  #This shuffles the starting locations without seperating the dataset from the start time.
  np.random.shuffle(arr)
  arr=np.array(arr)

  window_df_starts = list(arr[:,0])
  window_starts= list(arr[:,1])
  return window_starts,window_df_starts
#This array holds all the EEG datasets this program will be using

dfs=[]
for folder in folders_for_making_model:
  print(folder)
  os.chdir(folder)
  for x in os.listdir():
    if x.endswith("formated.csv"):
      print(x)
      # This appends all the csv files to the array
      dfs.append( pd.read_csv(str(x)))
  #Go back so that a new folder can be found
  os.chdir("..")

window_starts,window_df_starts= vectorMakerV2()

#These lines of code take the variables from the beginning, calculate how many windows will be needed, and prints some information


time_between_signals = 3.906266
#This window size generates a number of samplings in a window of specific length
number_of_windows=(len(window_starts))
window_time= (window_size*time_between_signals)/1000
validation_windows=int(number_of_windows*percent_validation)
testing_windows=   int(number_of_windows*percent_testing)
training_windows=  int(number_of_windows*percent_training)

print("the Period of interest is",Period_of_interest/3600,"hour\n",
      "there are",number_of_windows,'windows,\n'
      " This setup will generate:\n",training_windows , "windows for training,",percent_training,"%\n",
      validation_windows," windows for validation",percent_validation,"% &\n",
      testing_windows," Windows for testing,",percent_testing,"%.\n",
      "Each window has",window_time,"seconds, and",window_size,"timesteps\n",
      "This is about",number_of_windows*window_time/60/60,"hours of data\n")

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

#This function is the same as the last, but with the seizures excluded
def LabelVector2(label):
  zero_vector=[1,0]
  one_vector =[0,1]
  if label==0:
      return zero_vector
  else :
      return one_vector

def frameArrayMakerV4(window_starts, window_df_starts, normalize,fft, include_seizures):
  frames_array=[]
  label_array=[]
  label_OHV=[]

  for i in range(0,len(window_starts)):
    fft_array=[]
    start=window_starts[i]
    df=window_df_starts[i]
    #Still ordered data from a dataframe of a specific size
    new_df=(dfs[df]).loc[start:start+window_size-1]

  #Stores the labels in an array
    #If the number is negative, it is a seizure and is marked with a 2
    if (new_df.at[start+window_size-1,"Label"] <0):
      new_label=2
    #if the label is 0, it is after the last siezure, i don;t really know how to treat it
    elif (new_df.at[start+window_size-1,"Label"] ==0):
      new_label = 0
    #if the label is greater than the poi, then it is far from the seizure and is marked with a 0
    elif(new_df.at[start+window_size-1,"Label"] >Period_of_interest):
      new_label=0
    #If none of the others were true, it must be in the period of interest, and it gets marked with a 1
    else:
      new_label=1


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
      frames_array.append( new_df)

  return frames_array, label_array, label_OHV

#This generates the windows for training, testing, and validation.
#If hot vectors it true it will return your labels in the form of a a one-hot-vector
#If print ratio is true, it will print the number of each type of label and their ratio
#if normalize is true, it will perform z-score normalization
def get_dataset_partitions_tf(dfs, hot_vector=False, print_ratio = True, normalize=False, fft=False, include_seizures = True):
    print("fft:",fft,', normalize:',normalize,", include Seizures:",include_seizures)

    #For testing data
    testing_ds, testing_lables,testing_labels_OHV,=frameArrayMakerV4(window_starts[0:testing_windows]
                                                                     ,window_df_starts[0:testing_windows]
                                                                   ,normalize,fft,include_seizures)
    print('.',end='')
    #For validation data
    validation_ds, validation_lables,validatiion_labels_OHV,=frameArrayMakerV4(window_starts[testing_windows:testing_windows+validation_windows]
                                                                               ,window_df_starts[testing_windows:testing_windows+validation_windows]
                                                                               ,normalize,fft,include_seizures)
    print('.',end='')
    #for training data
    X, y,training_labels_OHV =frameArrayMakerV4(window_starts[testing_windows+validation_windows+1:number_of_windows]
                                                ,window_df_starts[testing_windows+validation_windows+1:number_of_windows]
                                                ,normalize,fft,include_seizures)

    X=np.array(X)
    training_windows=len(X)
    y=np.array(y)

    print('.',end='')

    # summarize class distribution
    counter = Counter(y)
    print('.',end='')
    # define pipeline
    over = SMOTE(sampling_strategy=1)
    under = RandomUnderSampler(sampling_strategy=1)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    print('.')
    # transform the dataset
    X=X.reshape(training_windows,-1)
    X, y = pipeline.fit_resample(X, y)
    # summarize the new class distribution
    print(counter)
    counter = Counter(y)
    X=X.reshape(-1,600,number_of_channels)
    print(counter)

    return X, testing_ds, validation_ds, y, testing_lables, validation_lables

#This generates the windows for training, testing, and validation.
#If hot vectors it true it will return your labels in the form of a a one-hot-vector
train_ds, test_ds, valid_ds, train_lables, test_lables,  valid_lables = get_dataset_partitions_tf(dfs,print_ratio=True, normalize=True, include_seizures=False)

training_data = np.array(train_ds)
testing_data = np.array(test_ds)
validation_data= np.array(valid_ds)


training_lables= np.array(train_lables)
testing_lables= np.array(test_lables)
validation_lables=np.array(valid_lables)


print("training  ",training_data.shape,",",training_lables.shape)
print("testing   ",testing_data.shape,",",testing_lables.shape)
print("validating",validation_data.shape,",",validation_lables.shape)
print('\n')

"""# Models & Training"""

learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=.01,
                                                                first_decay_steps=20, t_mul=2.0, m_mul=1.0, alpha=0.0, name=None)
opt = keras.optimizers.legacy.Adam(learning_rate=learning_rate)

# normalized .5466,
input_shape = ( window_size, number_of_channels, 1)
training_data = training_data.reshape(-1,window_size, number_of_channels, 1)

print("training  ",training_data.shape,",",training_lables.shape)
m2Dlstm = keras.Sequential([
  layers.Conv2D(filters=32, kernel_size=(5,5) ,input_shape=input_shape),
  layers.Conv2D(filters=16, kernel_size=(3,3)),
  layers.MaxPool2D( pool_size=(11, 5)),
  layers.Reshape((54,32)),
  layers.LSTM(90, activation='relu' ),
  layers.Dense(75),
  layers.Dense(40),
  layers.Dense(1)
])
print('\n',m2Dlstm.summary())
m2Dlstm.compile(
              optimizer='Adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

m2Dlstm.fit(training_data, training_lables, batch_size=batch_size, epochs=number_of_epochs, validation_data=(validation_data, validation_lables))

"""# m1DLSTM

"""

input_shape = ( window_size, number_of_channels, 1)
training_data = training_data.reshape(training_windows,-1, number_of_channels,1)

m1DLSTM = keras.Sequential([
  layers.Conv1D(filters=64, kernel_size=3, input_shape=input_shape),
  layers.Conv1D(filters=64, kernel_size=3),
  layers.MaxPool2D(),
  layers.Reshape([300,448]),
  layers.LSTM(100, activation='relu' ),
  layers.Dense(100),
  layers.Dense(50),
  layers.Dense(1)

])
print(m1DLSTM.summary())
m1DLSTM.compile(
              optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
m1DLSTM.fit(training_data, training_lables, batch_size=batch_size, epochs=number_of_epochs, validation_data=(validation_data, validation_lables))

input_shape = ( window_size, number_of_channels, 1)
training_data = training_data.reshape(training_windows,window_size, number_of_channels, 1)

m2Dconv = keras.Sequential([
  layers.Conv2D(filters=32, kernel_size=(5,5) ,input_shape=input_shape),
  layers.Conv2D(filters=16, kernel_size=(3,3)),
  layers.MaxPool2D( pool_size=(50, 5)),
  layers.Flatten(),
  layers.Dense(100),
  layers.Dense(1)
])
print(m2Dconv.summary())
m2Dconv.compile(
              optimizer='Adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
#normalized:.4955 , non-normalized: 0.4955
m2Dconv.fit(training_data, training_lables, batch_size=batch_size, epochs=number_of_epochs, validation_data=(validation_data, validation_lables))

print('\n--Accuracy on testing data--')
print('m2Dlstm')
test_loss, test_acc = m2Dlstm.evaluate(testing_data,testing_lables, verbose=2)
print( test_acc)

print('m1DLSTM')
test_loss, test_acc = m1DLSTM.evaluate(testing_data,testing_lables, verbose=2)
print( test_acc)

print('m2Dconv')
test_loss, test_acc = m2Dconv.evaluate(testing_data,testing_lables, verbose=2)
print( test_acc)