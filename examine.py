# -*- coding: utf-8 -*-
"""examine

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16r5zqDZhs6Q27-8gj6nwX1vLMr-YxGGq
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python import training
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import time
import csv
import os

class My_Custom_Generator(keras.utils.Sequence) :

  def __init__(self, filenames, labels, batch_size) :
    self.filenames = filenames
    self.labels = labels
    self.batch_size = batch_size


  def __len__(self) :
    return int(np.ceil(len(self.filenames) // float(self.batch_size)))


  def __getitem__(self, idx) :
    os.chdir('/content/drive/MyDrive/ERD/files')
    batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    return np.array([np.memmap(str(file_name), dtype='float32', mode='r', shape=(600,18))
               for file_name in batch_x]), np.array(batch_y)

def make_dataset(subjects):
  data_files=[]
  data_labels=[]
  for subject in subjects:
    labels=np.load(subject+'data_labels.npy')
    files=np.load(subject+'data_files.npy')
    data_files=np.append(data_files,files)
    data_labels=np.append(data_labels,labels)
  print(data_labels.shape)
  data_files,vailidation_data,data_labels,vailidation_labels=train_test_split(data_files,data_labels, test_size=0.10)
  training_data,testing_data,training_labels,testing_labels=train_test_split(data_files,data_labels, test_size=0.10)

  return training_data,training_labels,testing_data,testing_labels,vailidation_data,vailidation_labels

np_files=['chb01',]
batch_size=4
os.chdir('/content/drive/MyDrive/ERD')
training_data,training_labels,testing_data,testing_labels,vailidation_data,vailidation_labels= make_dataset(np_files)
os.chdir('/content/drive/MyDrive/ERD/files')
my_training_generator = My_Custom_Generator(training_data, training_labels, batch_size)
an_item=my_training_generator[1]
print("training_files",training_data.shape)
print("training_labels",training_labels.shape)
print(an_item)
print("-----------------")
print(an_item[1])
print("-----------------")
print(an_item[0][1][1])
print("-----------------")
print(an_item[1].shape)