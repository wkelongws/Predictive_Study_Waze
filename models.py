
# coding: utf-8

# In[1]:

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
# get_ipython().magic(u'matplotlib inline')


# In[2]:

import pickle
(samples,labels) = pickle.load( open( "data_new.p", "rb" ) )


print(samples.shape)
print(labels.shape)


# In[3]:

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
samples, samples_rest, labels, labels_rest = train_test_split(samples, labels, test_size=0.3, random_state=0)
samples_validation, samples_test, labels_validation, labels_test = train_test_split(samples_rest, labels_rest, test_size=0.5, random_state=0)

one_hot_labels = one_hot(labels)
one_hot_labels_validation = one_hot(labels_validation)
one_hot_labels_test = one_hot(labels_test)

print(samples.shape)
print(one_hot_labels.shape)
print(samples_validation.shape)
print(one_hot_labels_validation.shape)
print(samples_test.shape)
print(one_hot_labels_test.shape)


# In[4]:

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


# In[5]:

def feature_select(samples_scaled,features):
    return samples_scaled[:,:,:,features]

features = [0,1,2,7,8]
samples_scaled = feature_select(samples_scaled,features)
samples_scaled_validation = feature_select(samples_scaled_validation,features)
samples_scaled_test = feature_select(samples_scaled_test,features)
print(samples_scaled.shape)
print(samples_scaled_validation.shape)
print(samples_scaled_test.shape)


# In[6]:

from keras.models import Sequential
from keras.layers import Reshape, Dense, Convolution2D, Deconvolution2D, Flatten, Input, Dropout, MaxPooling2D, Activation
from keras.models import model_from_json
from keras.activations import relu, softmax, linear
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU


# In[19]:

# deep CNN model

model = Sequential()
# 10*5*5
model.add(Convolution2D(16, 3, 3,input_shape=(10, 5, 5),border_mode='valid'))
model.add(ELU())
# 8*3*16
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(ELU())
# 6*1*32
model.add(Flatten())
# 192
model.add(Dense(64))
model.add(ELU())
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

history_object = model.fit(samples_scaled, one_hot_labels,                            validation_data=(samples_scaled_validation, one_hot_labels_validation),                            nb_epoch=100, batch_size=64, verbose=2)

### plot the training and validation loss for each epoch
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
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

fig.savefig('train_history1.png', bbox_inches='tight')


from keras.models import load_model
model.save('model1.h5')





