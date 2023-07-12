# -*- coding: utf-8 -*-
"""Smote

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CdlTWl52cnmg36xcGgw9L_h5nQNvHtyZ
"""

import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.datasets import make_classification

testing_windows=23023
validation_windows=23023
training_windows= 382190
window_size=600
number_of_channels=18

testing_df=np.memmap('testing_df', dtype='float32',shape=(testing_windows,window_size*number_of_channels) )
np.save('testing_data',testing_df)
testing_df = 0
os.remove('testing_df')

validation_df=np.memmap("validation_df", dtype='float32',shape=(validation_windows,window_size*number_of_channels))
np.save("validation_data",validation_df)
validation_df = 0
os.remove('validation_df')

training_lables=np.load("training_lables.npy")
training_data= np.memmap('training_df', mode='w+',dtype='float32',shape=(training_windows,window_size*number_of_channels))
print("training  ",training_data.shape,",",training_lables.shape)



# summarize class distribution
counter = Counter(training_lables)

# define pipeline
over = SMOTE(sampling_strategy=1)
under = RandomUnderSampler(sampling_strategy=1)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)


# transform the dataset
training_data, training_lables = pipeline.fit_resample(training_data, training_lables)
# summarize the new class distribution
print("before:\t",counter)
counter = Counter(training_lables)
print("after:\t",counter)
print("training  ",training_data.shape,",",training_lables.shape)

training_data=training_data.reshape(-1,window_size,number_of_channels)
np.save('training_data',training_data)
np.save("training_lables",training_lables)

os.remove('training_df')