# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:06:05 2018

@author: aitor
"""
from random import shuffle
import os

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

TRAIN_PATH = "./dataset/train/"
VAL_PATH = "./dataset/val/"
TEST_PATH = "./dataset/test/"

def build_resnet50_model():
    # get the model without the denses
    base_model = ResNet50(weights='imagenet', include_top='false')
    new_dense = base_model.output
    # add the new denses to classify the hate images
    new_dense = Dense(1024, activation='relu')(new_dense)
    predictions = Dense(2, activation='softmax')(new_dense)
    model = Model(inputs=base_model.input, outputs=predictions)
    # we will only train the new denses for the baseline
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ["accuracy"])
    return model

def get_training_data_resnet50():    
    filenames = os.listdir(TRAIN_PATH) 
    shuffle(filenames)
    X_train = []
    y_train = []
    total_hate = 0
    total_no_hate = 0
    print '*****TRAIN*****'
    for filename in filenames:
        if "no_hate" in filename:
            y_train.append([1,0])
            total_no_hate += 1
        else:
            y_train.append([0,1])
            total_hate += 1
        img = image.load_img(TRAIN_PATH+filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train.append(x)
    print 'Total train examples: ' + str(len(y_train))
    print 'No hate examples: ' + str(total_no_hate)
    print 'Hate examples: ' + str(total_hate)
        
    print '*****VAL*****'
    filenames = os.listdir(VAL_PATH) 
    shuffle(filenames)
    X_val = []
    y_val = []
    total_hate = 0
    total_no_hate = 0
    for filename in filenames:
        if "no_hate" in filename:
            y_val.append([1,0])
            total_no_hate += 1
        else:
            y_val.append([0,1])
            total_hate += 1
        img = image.load_img(VAL_PATH+filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_val.append(x)
    print 'Total val examples: ' + str(len(y_val))
    print 'No hate examples: ' + str(total_no_hate)
    print 'Hate examples: ' + str(total_hate)
    
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    print 'Starting'
    print 'Building model...'
    model = build_resnet50_model()
    model.summary()
    print 'Building data...'
    X_train, y_train, X_val, y_val = get_training_data_resnet50()
    print 'X_train: ' + str(len(X_train))
    print(X_train.shape)
    print 'y_train: ' + str(len(y_train))
    print(y_train.shape)
    print 'X_val: ' + str(len(X_val))
    print(X_val.shape)
    print 'y_val: ' + str(len(y_val))
    print(y_val.shape)
    print 'Training mode...'
    results = model.fit(X_train, y_train, epochs= 100, batch_size = 32, validation_data = (X_val, y_val))
    print 'fin'
