
# coding: utf-8

# In[ ]:

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import csv
from pylab import *
from matplotlib import gridspec
import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from matplotlib import gridspec
cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.125, 1.0, 1.0),
                 (0.25, 1.0, 1.0),
                 (0.5625, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5625, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)


# In[ ]:

import pickle
(samples,labels) = pickle.load( open( "data_train.p", "rb" ) )
(samples_test,labels_test) = pickle.load( open( "data_test.p", "rb" ) )
print(samples.shape)
print(labels.shape)
print(samples_test.shape)
print(labels_test.shape)


# In[ ]:

from sklearn.utils import shuffle
def class_balance(samples,labels):
    samples0 = samples[labels==0,:,:,:]
    samples1 = samples[labels==1,:,:,:]
    selected = np.random.choice(len(samples0), len(samples1), replace=False)
    samples0 = samples0[selected,:,:,:]
    labels0 = np.zeros((len(samples0)))
    labels1 = np.ones((len(samples1)))
    samples = np.concatenate((samples0,samples1),axis=0)
    labels = np.concatenate((labels0,labels1),axis=0)
    samples, labels = shuffle(samples, labels)
    return samples, labels
def one_hot(labels):
    one_hot_labels = np.zeros((labels.shape[0],2))
    one_hot_labels[labels==0,0]=1
    one_hot_labels[labels==1,1]=1
    return one_hot_labels


samples, labels = class_balance(samples,labels)

from sklearn.model_selection import train_test_split
samples, samples_validation, labels, labels_validation = train_test_split(samples, labels, test_size=0.2, random_state=0)


one_hot_labels = one_hot(labels)
one_hot_labels_validation = one_hot(labels_validation)
one_hot_labels_test = one_hot(labels_test)

print(samples.shape)
print(one_hot_labels.shape)
print(samples_validation.shape)
print(one_hot_labels_validation.shape)
print(samples_test.shape)
print(one_hot_labels_test.shape)


# In[ ]:

def MinMax_Normalization(samples):
    samples_shape = samples.shape
    samples = np.reshape(samples,(samples_shape[0],samples_shape[1]*samples_shape[2]*samples_shape[3]))
    scaler = MinMaxScaler().fit(samples)
    samples_normalized = scaler.transform(samples)
    samples_normalized = np.reshape(samples_normalized,(samples_shape[0],samples_shape[1],samples_shape[2],samples_shape[3]))
    return samples_normalized, scaler

def transfer_scale(samples,scaler):
    samples_shape = samples.shape
    samples = np.reshape(samples,(samples_shape[0],samples_shape[1]*samples_shape[2]*samples_shape[3]))
    samples_normalized = scaler.transform(samples)
    samples_normalized = np.reshape(samples_normalized,(samples_shape[0],samples_shape[1],samples_shape[2],samples_shape[3]))
    return samples_normalized
    
samples_scaled, scaler = MinMax_Normalization(samples)
samples_scaled_validation = transfer_scale(samples_validation,scaler)
samples_scaled_test = transfer_scale(samples_test,scaler)

print(samples.shape)
print(one_hot_labels.shape)
print(samples_validation.shape)
print(one_hot_labels_validation.shape)
print(samples_test.shape)
print(one_hot_labels_test.shape)


# In[ ]:

def feature_select(samples_scaled,features):
    return samples_scaled[:,:,:,features]

features = [0,1,2,3,4,5,6,7,8]
samples_scaled = feature_select(samples_scaled,features)
samples_scaled_validation = feature_select(samples_scaled_validation,features)
samples_scaled_test = feature_select(samples_scaled_test,features)
print(samples_scaled.shape)
print(samples_scaled_validation.shape)
print(samples_scaled_test.shape)


# In[ ]:

from keras.models import Sequential
from keras.layers import Reshape, Dense, Convolution2D, Deconvolution2D, Flatten, Input, Dropout, MaxPooling2D, Activation
from keras.models import model_from_json
from keras.activations import relu, softmax, linear
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU

### plot the training and validation loss for each epoch
def history_plot(history_object,image):
  
    index = np.where(np.array(history_object.history['val_acc']) == np.max(history_object.history['val_acc']))
    train_acc = history_object.history['acc'][index[0][0]]
    valid_acc = history_object.history['val_acc'][index[0][0]]
    best_epoch = history_object.epoch[index[0][0]]+1
    fig = plt.figure(figsize=(15,11))
    gs = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
    ax0 = plt.subplot(gs[0])
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    ax1 = plt.subplot(gs[1])
    plt.plot(history_object.history['acc'])
    plt.plot(history_object.history['val_acc'])
    plt.title('model accuracy: train = ' + str(train_acc) + ' val = ' + str(valid_acc) + ' observed at: Epoch = ' + str(best_epoch))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    fig.savefig(image, bbox_inches='tight')

# In[ ]:

# deep CNN model

## CNN_ReLU_BNno_DROPno_2k
#epochs = 2000
#batch_size = 4096
#rate = 0
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
##model.add(BatchNormalization())
#model.add(Activation('relu'))
##model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
##model.add(BatchNormalization())
#model.add(Activation('relu'))
##model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
##model.add(BatchNormalization())
#model.add(Activation('relu'))
##model.add(ELU())
#model.add(Dropout(rate))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ReLU_BNno_DROPno_2k.png')
#
#
#
## CNN_ELU_BNno_DROPno_2k
#epochs = 2000
#batch_size = 4096
#rate = 0
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
##model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
##model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
##model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ELU_BNno_DROPno_2k.png')
#
#
## CNN_ELU_BNyes_DROPno_2k
#epochs = 2000
#batch_size = 4096
#rate = 0
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ELU_BNyes_DROPno_2k.png')
#
## CNN_ReLU_BNyes_DROPno_2k
#epochs = 2000
#batch_size = 4096
#rate = 0
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
##model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
##model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
##model.add(ELU())
#model.add(Dropout(rate))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ReLU_BNyes_DROPno_2k.png')
#
## CNN_ELU_BNyes_DROPlast0.25_2k
#epochs = 2000
#batch_size = 4096
#rate = 0
#rate_last = 0.25
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate_last))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ELU_BNyes_DROPlast0.25_2k.png')
#
## CNN_ELU_BNyes_DROPlast0.5_2k
#epochs = 2000
#batch_size = 4096
#rate = 0
#rate_last = 0.5
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate_last))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ELU_BNyes_DROPlast0.5_2k.png')
#
#
## CNN_ELU_BNyes_DROPall0.25_2k
#epochs = 2000
#batch_size = 4096
#rate = 0.25
#rate_last = 0.25
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate_last))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ELU_BNyes_DROPall0.25_2k.png')
#
#
## CNN_ELU_BNyes_DROPall0.5_2k
#epochs = 2000
#batch_size = 4096
#rate = 0.5
#rate_last = 0.5
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate_last))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ELU_BNyes_DROPall0.5_2k.png')
#
## CNN_ReLU_BNyes_DROPall0.25_2k
#epochs = 2000
#batch_size = 4096
#rate = 0.25
#rate_last = 0.25
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
##model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
##model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
##model.add(ELU())
#model.add(Dropout(rate_last))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ReLU_BNyes_DROPall0.25_2k.png')
#
#
## CNN_ELU_BNyes_DROPall0.5_10k
#epochs = 10000
#batch_size = 4096
#rate = 0.5
#rate_last = 0.5
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate_last))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ELU_BNyes_DROPall0.5_10k.png')
#
## CNN_ELU_BNyes_DROPall0.25_10k
#epochs = 10000
#batch_size = 4096
#rate = 0.25
#rate_last = 0.25
#model = Sequential()
## 10*5*5
#model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 8*3*16
#model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate))
## 6*1*32
#model.add(Flatten())
## 192
#model.add(Dense(64))
#model.add(BatchNormalization())
##model.add(Activation('relu'))
#model.add(ELU())
#model.add(Dropout(rate_last))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
#history_plot(history_object,'CNN_ELU_BNyes_DROPall0.25_10k.png')

# CNN_ReLU_BNyes_DROPall0.25_10k
epochs = 10000
batch_size = 4096
rate = 0.25
rate_last = 0.25
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate_last))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNN_ReLU_BNyes_DROPall0.25_10k.png')



# small size CNN

# CNNsmall_ReLU_BNno_DROPno_2k
epochs = 10000
batch_size = 4096
rate = 0
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ReLU_BNno_DROPno_2k.png')



# CNNsmall_ELU_BNno_DROPno_2k
epochs = 10000
batch_size = 4096
rate = 0
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ELU_BNno_DROPno_2k.png')


# CNNsmall_ELU_BNyes_DROPno_2k
epochs = 10000
batch_size = 4096
rate = 0
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ELU_BNyes_DROPno_2k.png')

# CNNsmall_ReLU_BNyes_DROPno_2k
epochs = 10000
batch_size = 4096
rate = 0
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ReLU_BNyes_DROPno_2k.png')

# CNNsmall_ELU_BNyes_DROPlast0.25_2k
epochs = 10000
batch_size = 4096
rate = 0
rate_last = 0.25
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate_last))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ELU_BNyes_DROPlast0.25_2k.png')

# CNNsmall_ELU_BNyes_DROPlast0.5_2k
epochs = 10000
batch_size = 4096
rate = 0
rate_last = 0.5
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate_last))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ELU_BNyes_DROPlast0.5_2k.png')


# CNNsmall_ELU_BNyes_DROPall0.25_2k
epochs = 10000
batch_size = 4096
rate = 0.25
rate_last = 0.25
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate_last))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ELU_BNyes_DROPall0.25_2k.png')


# CNNsmall_ELU_BNyes_DROPall0.5_2k
epochs = 10000
batch_size = 4096
rate = 0.5
rate_last = 0.5
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate_last))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ELU_BNyes_DROPall0.5_2k.png')

# CNNsmall_ReLU_BNyes_DROPall0.25_2k
epochs = 10000
batch_size = 4096
rate = 0.25
rate_last = 0.25
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate_last))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ReLU_BNyes_DROPall0.25_2k.png')


# CNNsmall_ELU_BNyes_DROPall0.5_10k
epochs = 10000
batch_size = 4096
rate = 0.5
rate_last = 0.5
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate_last))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ELU_BNyes_DROPall0.5_10k.png')

# CNNsmall_ELU_BNyes_DROPall0.25_10k
epochs = 10000
batch_size = 4096
rate = 0.25
rate_last = 0.25
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate_last))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ELU_BNyes_DROPall0.25_10k.png')

# CNNsmall_ReLU_BNyes_DROPall0.25_10k
epochs = 10000
batch_size = 4096
rate = 0.25
rate_last = 0.25
model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 9),border_mode='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate_last))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'CNNsmall_ReLU_BNyes_DROPall0.25_10k.png')




# In[ ]:

# deep ANN model

# ANN_ReLU_BNno_DROPno_2k
epochs = 2000
batch_size = 4096
rate = 0
model = Sequential()
model.add(Flatten(input_shape=(10, 5, 9)))
# 192
model.add(Dense(128))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(64))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(32))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'ANN_ReLU_BNno_DROPno_2k.png')

# ANN_ELU_BNno_DROPno_2k
epochs = 2000
batch_size = 4096
rate = 0
model = Sequential()
model.add(Flatten(input_shape=(10, 5, 9)))
# 192
model.add(Dense(128))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(64))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(32))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'ANN_ELU_BNno_DROPno_2k.png')

# ANN_ELU_BNyes_DROPno_2k
epochs = 2000
batch_size = 4096
rate = 0
model = Sequential()
model.add(Flatten(input_shape=(10, 5, 9)))
# 192
model.add(Dense(128))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(64))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(32))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'ANN_ELU_BNyes_DROPno_2k.png')

# ANN_ELU_BNyes_DROPall0.25_2k
epochs = 2000
batch_size = 4096
rate = 0.25
model = Sequential()
model.add(Flatten(input_shape=(10, 5, 9)))
# 192
model.add(Dense(128))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(64))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(32))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'ANN_ELU_BNyes_DROPall0.25_2k.png')

# ANN_ELU_BNyes_DROPall0.5_2k
epochs = 2000
batch_size = 4096
rate = 0.5
model = Sequential()
model.add(Flatten(input_shape=(10, 5, 9)))
# 192
model.add(Dense(128))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(64))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(32))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'ANN_ELU_BNyes_DROPall0.5_2k.png')

# ANN_ELU_BNyes_DROPall0.5_10k
epochs = 10000
batch_size = 4096
rate = 0.5
model = Sequential()
model.add(Flatten(input_shape=(10, 5, 9)))
# 192
model.add(Dense(128))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(64))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(32))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'ANN_ELU_BNyes_DROPall0.5_10k.png')

# ANN_ELU_BNyes_DROPall0.25_10k
epochs = 10000
batch_size = 4096
rate = 0.25
model = Sequential()
model.add(Flatten(input_shape=(10, 5, 9)))
# 192
model.add(Dense(128))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(64))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(32))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dropout(rate))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=epochs, batch_size=batch_size, verbose=2)
history_plot(history_object,'ANN_ELU_BNyes_DROPall0.25_10k.png')

#from keras.models import load_model
#model.save('model.h5')

