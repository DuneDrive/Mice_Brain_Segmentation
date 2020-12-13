#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#import keras
import os
from fnmatch import fnmatch
import tensorflow as tf
import scipy.io as sio
import tensorflow.keras as keras
from keras import backend as K
from matplotlib import pyplot as plt
 
 
from sklearn.model_selection import KFold # tool for getting random folds in K-fold cross validation
from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Dropout, BatchNormalization, concatenate, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant
from keras.models import load_model
import h5py
import nibabel as nib

from tensorflow.python.client import device_lib 

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])






def dice_metric(y_true, y_pred):
#A Fuction that calculates the dice score for the predicted and true regions
#Inputs: The predicted and true labels of what in the image is skull
#Outputs: A dice score

    threshold = 0

    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)
    mask = y_true > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.multiply(y_true, mask)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    hard_dice = (2. * inse) / (l + r)

    hard_dice = tf.reduce_mean(hard_dice)

    return hard_dice


def dice_metric_thresh(y_true, y_pred, thresh):
#A Fuction that calculates the dice score for the predicted and true regions
#Inputs: The predicted and true labels of what in the image is skull
#Outputs: A dice score

    threshold = thresh

    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)
    #mask = y_true > threshold
    #mask = tf.cast(mask, dtype=tf.float32)
    #y_true = tf.multiply(y_true, mask)
    y_true = tf.cast(y_true, dtype=tf.float32)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    hard_dice = (2. * inse) / (l + r)

    hard_dice = tf.reduce_mean(hard_dice)

    return hard_dice






def cnn(fliter_num,kernel_size):
    with strategy.scope():
        #Layers to construct the model.
        #Each unit of the U-net consists of 2 convultional layers 1 pooling/unpooling layer
        #The kernel siez for these are 3x3 and 2x2 respectively
        #All layers pad to insure data dimensions stay constant before and after model
        #Skip connections connect pooling and upooling layers of the same data size using concatenation
        #All layers use rectiliar units, except the last layer which uses sigmoidal.
        #Rectilinear was chosen to effectively weed out unimportant data, and the sigmoidal was used because it looks like a percent
        #Email me any questions
        input_layer = keras.layers.Input(shape=(180, 100, 1))
        conv1a = keras.layers.Conv2D(filters=fliter_num, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(input_layer)
        conv1b = keras.layers.Conv2D(filters=fliter_num, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv1a)
        pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv1b)
        conv2a = keras.layers.Conv2D(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(pool1)
        conv2b = keras.layers.Conv2D(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv2a)
        pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2b)
        conv3a = keras.layers.Conv2D(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(pool2)
        conv3b = keras.layers.Conv2D(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv3a)
    
        dconv3a = keras.layers.Conv2DTranspose(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), padding='same')(conv3b)
        dconv3b = keras.layers.Conv2DTranspose(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), padding='same')(dconv3a)
        unpool2 = keras.layers.UpSampling2D(size=(2, 2))(dconv3b)
        cat2 = keras.layers.concatenate([conv2b, unpool2])
        dconv2a = keras.layers.Conv2DTranspose(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), padding='same')(cat2)
        dconv2b = keras.layers.Conv2DTranspose(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), padding='same')(dconv2a)
        unpool1 = keras.layers.UpSampling2D(size=(2, 2))(dconv2b)
        cat1 = keras.layers.concatenate([conv1b, unpool1])
        dconv1a = keras.layers.Conv2DTranspose(filters=fliter_num, kernel_size=(kernel_size, kernel_size), padding='same')(cat1)
        dconv1b = keras.layers.Conv2DTranspose(filters=fliter_num, kernel_size=(kernel_size, kernel_size), padding='same')(dconv1a)
    
        output = keras.layers.Conv2D(filters=1, kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')(dconv1b)
    
        #This saves our model every 10 epochs, just incase it is better than the final/we crash
        #cp_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(dirnam, "models/cp-{epoch:04d}.ckpt"), verbose=1, save_weights_only=True, period=10)
    
        #Compiling the model.  Learning rate turned down because experimentally it does better at this value
        #Cross entropy used rather than dice score because experimentally causes model to converge to local min faster
        model = keras.models.Model(inputs=input_layer, outputs=output)
        

        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=5e-5,
            decay_steps=200,
            decay_rate=0.9)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        
        
        
        
        
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[dice_metric])
        history = model.fit(x_train, y_train, epochs=100, batch_size=50, validation_data=(x_test, y_test))

        train_loss = history.history['loss']
        np.save('/home/alex/Downloads/Mouse-Segmentation-master/results/train_loss_' + str(i) + '.npy',train_loss)
        train_acc = history.history['dice_metric']
        np.save('/home/alex/Downloads/Mouse-Segmentation-master/results/train_acc_mm_' + str(i) + '.npy',train_acc)
        val_loss = history.history['val_loss']
        np.save('/home/alex/Downloads/Mouse-Segmentation-master/results/val_loss_' + str(i) + '.npy',val_loss)
        val_acc = history.history['val_dice_metric']
        np.save('/home/alex/Downloads/Mouse-Segmentation-master/results/val_acc_mm_' + str(i) + '.npy',val_acc)
        model.save('/home/alex/Downloads/Mouse-Segmentation-master/results/model_' + str(i) + '.h5')
        test_pred = model.predict(x_test)
        np.save('/home/alex/Downloads/Mouse-Segmentation-master/results/pred_' + str(i) + '.npy', test_pred)
        # save in nifti format
    

#model=cnn();




#def leave_one_out():
image = nib.load('/home/alex/Downloads/Mouse-Segmentation-master/mouse_full_data_120720.nii')
data = image.get_fdata()

for i in range(6,len(data)):
    
    x_train_pre = np.concatenate((data[:i,:,:,:,0], data[i+1:,:,:,:,0]), axis=0, out=None)
    x_train = x_train_pre.transpose(0,3,1,2).reshape(len(x_train_pre)*200,180,100,1)
    
    y_train_pre = np.concatenate((data[:i,:,:,:,1], data[i+1:,:,:,:,1]), axis=0, out=None)
    y_train = y_train_pre.transpose(0,3,1,2).reshape(len(y_train_pre)*200,180,100,1)
    
    
    x_test_pre = data[i,:,:,:,0]
    x_test = x_test_pre.transpose(2,0,1).reshape(200,180,100,1)
    
    y_test_pre = data[i,:,:,:,1]
    y_test = y_test_pre.transpose(2,0,1).reshape(200,180,100,1)
    cnn(32,5)
    


    





