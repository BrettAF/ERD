{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/BrettAF/ERD/blob/main/Windowing_and_training.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "i523q0gGIkz6"
      },
      "source": [
        "# Libraries and variables"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 39,
      "metadata": {
        "id": "G0z1pYHppqev"
      },
      "outputs": [],
      "source": [
        "number_of_windows = 25200\n",
        "window_size = 600\n",
        "percent_testing=.1\n",
        "percent_validation=.1\n",
        "batch_size = 128\n",
        "number_of_epochs=300\n",
        "\n",
        "#This is the period before a seizure that will be examined.\n",
        "Period_of_interest=60*60*1"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "rj5wqB8wYxXE"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "from pathlib import Path\n",
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from keras import layers\n",
        "from tensorflow.python import training\n",
        "\n",
        "# Make numpy values easier to read.\n",
        "np.set_printoptions(precision=3, suppress=True)\n",
        "import random as r\n",
        "import scipy as sp\n",
        "from scipy.stats import zscore\n",
        "from scipy.fft import fft, ifft\n",
        "\n",
        "from keras.layers.rnn import LSTM"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 40,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CrT5xDgpyHT0",
        "outputId": "ca3a8b46-540d-47b1-ed6a-ebc130406a88"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "the Period of interest is 1.0 hour\n",
            " This setup will generate:\n",
            " 20160  windows for training,\n",
            " 2520 windows for validation &\n",
            " 2520 Windows for testing.\n",
            " Each window has 2.3437596 seconds, and 600 frames\n",
            " This is about 16.4063172 hours of data\n"
          ]
        }
      ],
      "source": [
        "\n",
        "time_between_signals = 3.906266\n",
        "#This window size generates a number of samplings in a window of specific length\n",
        "window_time= (window_size*time_between_signals)/1000\n",
        "\n",
        "validation_windows=int(number_of_windows*(percent_validation))\n",
        "testing_windows=int(number_of_windows*(percent_testing))\n",
        "training_windows=number_of_windows-testing_windows-validation_windows\n",
        "\n",
        "print(\"the Period of interest is\",Period_of_interest/3600,\"hour\\n\",\n",
        "      \"This setup will generate:\\n\",training_windows , \" windows for training,\\n\",\n",
        "      validation_windows,\"windows for validation &\\n\",\n",
        "      testing_windows,\"Windows for testing.\\n\",\n",
        "      \"Each window has\",window_time,\"seconds, and\",window_size,\"frames\\n\",\n",
        "      \"This is about\",number_of_windows*window_time/60/60,\"hours of data\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IW6Jw_M9JQSt"
      },
      "source": [
        "# Reading & Shaping Data"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eIoBoVKWcUGk",
        "outputId": "5cfb0f6b-a777-4f50-b2f3-1bf21374ab73"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "chb01_03 formated.csv\n",
            "chb01_01 formated.csv\n"
          ]
        }
      ],
      "source": [
        "dfs=[]\n",
        "for x in os.listdir():\n",
        "    if x.endswith(\"formated.csv\"):\n",
        "      print(x)\n",
        "      dfs.append( pd.read_csv(str(x)))\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#for recording the lables of the dataset as a one-hot vector\n",
        "def LabelVector3(label):\n",
        "  zero_vector=[1,0,0]\n",
        "  one_vector =[0,1,0]\n",
        "  two_vector =[0,0,1]\n",
        "\n",
        "  if label==0:\n",
        "      return zero_vector\n",
        "  elif label ==1:\n",
        "      return one_vector\n",
        "  else: return two_vector\n",
        "\n",
        "def LabelVector2(label):\n",
        "  zero_vector=[1,0]\n",
        "  one_vector =[0,1]\n",
        "  if label==0:\n",
        "      return zero_vector\n",
        "  else :\n",
        "      return one_vector\n",
        "\n",
        "#This ensures that no two training windows are the same.\n",
        "#It generates a vector of shuffled unique start locations\n",
        "def vectorMakerV2():\n",
        "  arr=[]\n",
        "  for df in range(0,len(dfs)):\n",
        "    for row in range(0,len(dfs[df])-window_size-1,window_size):\n",
        "        arr.append([df,row])\n",
        "  np.random.shuffle(arr)\n",
        "  arr=np.array(arr)\n",
        "\n",
        "  window_df_starts = list(arr[:,0])\n",
        "  window_starts= list(arr[:,1])\n",
        "  return window_starts,window_df_starts"
      ],
      "metadata": {
        "id": "fPQ-8Nv_fgpK"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def frameArrayMakerV3(i, window_length,  window_starts, window_df_starts,normalize,fft_flag, include_seizures):\n",
        "  frames_array=[]\n",
        "  label_array=[]\n",
        "  label_OHV=[]\n",
        "\n",
        "  while i<window_length:\n",
        "    fft_array=[]\n",
        "    start=window_starts[i]\n",
        "    df=window_df_starts[i]\n",
        "    i=i+1\n",
        "    #Still ordered data from a dataframe of a specific size\n",
        "    new_df=(dfs[df]).loc[start:start+window_size-1]\n",
        "\n",
        "  #Stores the labels in an array\n",
        "    #If the number is negative, it is a seizure and is marked with a 2\n",
        "    if (new_df.at[start+window_size-1,\"Label\"] <0):\n",
        "      new_label=2\n",
        "    #if the label is 0, it is after the last siezure, i don;t really know how to treat it\n",
        "    elif (new_df.at[start+window_size-1,\"Label\"] ==0):\n",
        "      new_label = 0\n",
        "    #if the label is greater than the poi, then it is far from the seizure and is marked with a 0\n",
        "    elif(new_df.at[start+window_size-1,\"Label\"] >Period_of_interest):\n",
        "      new_label=0\n",
        "    #If none of the others were true, it must be in the period of interest, and it gets marked with a 1\n",
        "    else:\n",
        "      new_label=1\n",
        "\n",
        "\n",
        "    if ((include_seizures == True) or (new_label != 2) ):\n",
        "      new_df=new_df.drop(columns=[\"EEG Time\", \"Label\"])\n",
        "      # normalizes the columns\n",
        "      if normalize == True:\n",
        "          new_df = zscore(new_df)\n",
        "      label_array.append(new_label)\n",
        "\n",
        "      if (include_seizures == True):\n",
        "          label_OHV.append( LabelVector3(new_label))\n",
        "      else:\n",
        "          label_OHV.append( LabelVector2(new_label))\n",
        "\n",
        "        #if fft_flag == True:\n",
        "          #for j in range(0,16):\n",
        "            #one_channel=new_df[:,j]\n",
        "            #anfft=fft(one_channel)\n",
        "            #fft_array.append(anfft.real[20:window_size//2])\n",
        "            #fft_array.append(anfft.imag[20:window_size//2])\n",
        "          #new_df=fft_array\n",
        "      frames_array.append( new_df)\n",
        "\n",
        "\n",
        "  return frames_array, label_array, label_OHV,i"
      ],
      "metadata": {
        "id": "eLJhqxJOqBKL"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#This function guarantees that the number of each training label should stay aproximately even\n",
        "def rebalanceRatio(training_ds_p, training_lables_p,training_labels_OHV_p):\n",
        "\n",
        "  training_ds=training_ds_p[0:training_windows]\n",
        "  training_lables=training_lables_p[0:training_windows]\n",
        "  training_labels_OHV=training_labels_OHV_p[0:training_windows]\n",
        "\n",
        "  ratio=training_lables.count(1)/training_windows\n",
        "  print(\"ratio before adjusting:\",ratio)\n",
        "  #a moves through the tail\n",
        "  #b moves through the training window\n",
        "\n",
        "  a=training_windows\n",
        "  b=0\n",
        "  while ratio >.505 and(a<len(training_ds_p)-1)and(b<training_windows-1):\n",
        "    while training_lables_p[a] !=0:\n",
        "      a+=1\n",
        "    while training_lables[b] !=1:\n",
        "      b+=1\n",
        "    training_lables[b]=training_lables_p[a]\n",
        "    training_ds[b]=training_ds_p[a]\n",
        "    training_labels_OHV[b]=training_labels_OHV_p[a]\n",
        "    a+=1\n",
        "    b+=1\n",
        "    ratio=training_lables.count(1)/training_windows\n",
        "\n",
        "  while ratio <.495 and(a<len(training_ds_p)-1)and(b<training_windows-1):\n",
        "    while training_lables_p[a] !=1:\n",
        "      a+=1\n",
        "    while training_lables[b] !=0:\n",
        "      b+=1\n",
        "    training_lables[b]=training_lables_p[a]\n",
        "    training_ds[b]=training_ds_p[a]\n",
        "    training_labels_OHV[b]=training_labels_OHV_p[a]\n",
        "    a+=1\n",
        "    b+=1\n",
        "    ratio=training_lables.count(1)/training_windows\n",
        "\n",
        "  return training_ds,training_lables,training_labels_OHV\n"
      ],
      "metadata": {
        "id": "k_nIm0Vu2zYO"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "E-arwD2hDRQN"
      },
      "outputs": [],
      "source": [
        "#This generates the windows for training, testing, and validation.\n",
        "#If hot vectors it true it will return your labels in the form of a a one-hot-vector\n",
        "#If print ratio is true, it will print the number of each type of label and their ratio\n",
        "#if normalize is true, it will perform z-score normalization\n",
        "def get_dataset_partitions_tf(dfs, hot_vector=True, print_ratio = True, normalize=False, fft=False, include_seizures = True):\n",
        "\n",
        "    window_starts,window_df_starts= vectorMakerV2()\n",
        "\n",
        "    #For testing data\n",
        "    testing_ds, testing_lables,testing_labels_OHV,i =frameArrayMakerV3(0,testing_windows+1,  window_starts,window_df_starts,normalize,fft,include_seizures)\n",
        "\n",
        "    #For validation data\n",
        "    validation_ds, validation_lables,validatiion_labels_OHV,i =frameArrayMakerV3(i,validation_windows+testing_windows+1,  window_starts,window_df_starts,normalize,fft,include_seizures)\n",
        "\n",
        "     #for training data\n",
        "    training_ds_p, training_lables_p,training_labels_OHV_p,i =frameArrayMakerV3(i,len(window_starts), window_starts,window_df_starts,normalize,fft,include_seizures)\n",
        "    #print(i,window_starts[30],window_df_starts[30])\n",
        "    print('--Training Data--')\n",
        "\n",
        "    training_ds, training_lables,training_labels_OHV = rebalanceRatio(training_ds_p, training_lables_p,training_labels_OHV_p)\n",
        "    #0: not poi or seizure\n",
        "    #1: period of interest\n",
        "    #2: seizure\n",
        "\n",
        "\n",
        "\n",
        "    if print_ratio == True:\n",
        "      print(\"# of POI     :\",training_lables.count(1),\",\",100*training_lables.count(1)/training_windows,'%')\n",
        "      print(\"# of not POI :\",training_lables.count(0),\",\",100*training_lables.count(0)/training_windows,'%')\n",
        "      print(\"# of seizures:\",training_lables.count(2),\",\",100*training_lables.count(2)/training_windows,'%','\\n')\n",
        "    if hot_vector == True:\n",
        "      return training_ds, testing_ds, validation_ds, training_labels_OHV, testing_labels_OHV,validatiion_labels_OHV\n",
        "    return training_ds, testing_ds, validation_ds, training_lables, testing_lables, validation_lables"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XVb33bhaDZ9k",
        "outputId": "857c3e33-0ab4-4fa4-b4ba-ba633a499717"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "--Training Data--\n",
            "ratio before adjusting: 0.49166666666666664\n",
            "# of POI     : 713 , 49.513888888888886 %\n",
            "# of not POI : 727 , 50.486111111111114 %\n",
            "# of seizures: 0 , 0.0 % \n",
            "\n"
          ]
        }
      ],
      "source": [
        "#This generates the windows for training, testing, and validation.\n",
        "#If hot vectors it true it will return your labels in the form of a a one-hot-vector\n",
        "train_ds, test_ds, valid_ds, train_lables, test_lables,  valid_lables = get_dataset_partitions_tf(dfs,print_ratio=True, normalize=True, include_seizures=False, hot_vector=True)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_cwqnqBxuElK",
        "outputId": "45ddb162-55b2-4d92-b638-67d161a8df10"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "training   (1440, 600, 16) , (1440, 2)\n",
            "testing    (180, 600, 16) , (180, 2)\n",
            "validating (178, 600, 16) , (178, 2)\n",
            "\n",
            "\n"
          ]
        }
      ],
      "source": [
        "\n",
        "training_data = np.array(train_ds)\n",
        "testing_data = np.array(test_ds)\n",
        "validation_data= np.array(valid_ds)\n",
        "\n",
        "\n",
        "training_lables= np.array(train_lables)\n",
        "testing_lables= np.array(test_lables)\n",
        "validation_lables=np.array(valid_lables)\n",
        "\n",
        "\n",
        "print(\"training  \",training_data.shape,\",\",training_lables.shape)\n",
        "print(\"testing   \",testing_data.shape,\",\",testing_lables.shape)\n",
        "print(\"validating\",validation_data.shape,\",\",validation_lables.shape)\n",
        "print('\\n')\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pBPLLUY4JHV4"
      },
      "source": [
        "# Models & Training"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=.01,\n",
        "                                                                first_decay_steps=20, t_mul=2.0, m_mul=1.0, alpha=0.0, name=None)\n",
        "opt = keras.optimizers.legacy.Adam(learning_rate=learning_rate)"
      ],
      "metadata": {
        "id": "McgVKoft7hkZ"
      },
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# normalized .5466,\n",
        "input_shape = ( window_size, 16, 1)\n",
        "training_data = training_data.reshape(-1,window_size, 16, 1)\n",
        "\n",
        "print(\"training  \",training_data.shape,\",\",training_lables.shape)\n",
        "m2Dlstm = keras.Sequential([\n",
        "  layers.Conv2D(filters=32, kernel_size=(5,5) ,input_shape=input_shape),\n",
        "  layers.Conv2D(filters=16, kernel_size=(3,3)),\n",
        "  layers.MaxPool2D( pool_size=(11, 5)),\n",
        "  layers.Reshape((54,32)),\n",
        "  layers.LSTM(90, activation='relu' ),\n",
        "  layers.Dense(75),\n",
        "  layers.Dense(40),\n",
        "  layers.Dense(training_lables.shape[1])\n",
        "])\n",
        "print('\\n',m2Dlstm.summary())\n",
        "m2Dlstm.compile(\n",
        "              optimizer=opt,\n",
        "              loss=tf.keras.losses.CategoricalCrossentropy(),\n",
        "              metrics=['accuracy'])\n",
        "\n",
        "m2Dlstm.fit(training_data, training_lables, batch_size=batch_size, epochs=number_of_epochs, validation_data=(validation_data, validation_lables))"
      ],
      "metadata": {
        "id": "Jm98qfLlC2SG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xJlePCZ5xNs-"
      },
      "source": [
        "# Evaluating\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cuTp-AhExRT1",
        "outputId": "727b1912-2924-4b62-8e67-1ebb6283251f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "--Accuracy on testing data--\n",
            "6/6 - 0s - loss: 0.6828 - accuracy: 0.5556 - 387ms/epoch - 64ms/step\n",
            "0.5555555820465088\n"
          ]
        }
      ],
      "source": [
        "print('\\n--Accuracy on testing data--')\n",
        "test_loss, test_acc = m2Dlstm.evaluate(testing_data,testing_lables, verbose=2)\n",
        "print( test_acc)\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "m2Dlstm.save('.')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kpPZL2PgGhOA",
        "outputId": "7b0c8559-ae2d-4853-90bd-777d5aed2cd3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 3 of 3). These functions will not be directly callable after loading.\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}