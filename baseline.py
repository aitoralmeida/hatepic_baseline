# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:06:05 2018

@author: aitor
"""
from random import shuffle
import sys
import os

#from keras.applications.resnet50 import ResNet50
import keras.applications.inception_v3 as inceptionv3
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input
import numpy as np
from sklearn import metrics

TRAIN_PATH = "./dataset/train/"
VAL_PATH = "./dataset/val/"
TEST_PATH = "./dataset/test/"

'''
********************************************************
*******************RESNET50*****************************
********************************************************
'''
def build_resnet50_model():
    # get the model without the denses
    base_model = resnet50.ResNet50(weights='imagenet', include_top='false')
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
        x = resnet50.preprocess_input(x)
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
        x = resnet50.preprocess_input(x)
        X_val.append(x)
    print 'Total val examples: ' + str(len(y_val))
    print 'No hate examples: ' + str(total_no_hate)
    print 'Hate examples: ' + str(total_hate)  
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    return X_train, y_train, X_val, y_val

def get_test_data_resnet50(): 
    print '*****TEST*****'
    filenames = os.listdir(TEST_PATH) 
    shuffle(filenames)
    X_test = []
    y_test = []
    total_hate = 0
    total_no_hate = 0
    for filename in filenames:
        if "no_hate" in filename:
            y_test.append(0)
            total_no_hate += 1
        else:
            y_test.append(1)
            total_hate += 1
        img = image.load_img(TEST_PATH+filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = resnet50.preprocess_input(x)
        X_test.append(x)
    print 'Total val examples: ' + str(len(y_test))
    print 'No hate examples: ' + str(total_no_hate)
    print 'Hate examples: ' + str(total_hate)  
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_test, y_test

'''
********************************************************
*******************XCEPTION*****************************
********************************************************
'''
def build_xception_model():
    # get the model without the denses
    base_model = xception.Xception(weights='imagenet', include_top='false')
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

def get_training_data_xception():    
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
        x = xception.preprocess_input(x)
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
        x = xception.preprocess_input(x)
        X_val.append(x)
    print 'Total val examples: ' + str(len(y_val))
    print 'No hate examples: ' + str(total_no_hate)
    print 'Hate examples: ' + str(total_hate)  
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    return X_train, y_train, X_val, y_val

def get_test_data_xception(): 
    print '*****TEST*****'
    filenames = os.listdir(TEST_PATH) 
    shuffle(filenames)
    X_test = []
    y_test = []
    total_hate = 0
    total_no_hate = 0
    for filename in filenames:
        if "no_hate" in filename:
            y_test.append(0)
            total_no_hate += 1
        else:
            y_test.append(1)
            total_hate += 1
        img = image.load_img(TEST_PATH+filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = xception.preprocess_input(x)
        X_test.append(x)
    print 'Total val examples: ' + str(len(y_test))
    print 'No hate examples: ' + str(total_no_hate)
    print 'Hate examples: ' + str(total_hate)  
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_test, y_test

'''
********************************************************
*******************INCEPTIONV3**************************
********************************************************
'''
def build_inceptionv3_model():
    # get the model without the denses
    base_model = inceptionv3.InceptionV3(weights='imagenet', include_top='false')
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

def get_training_data_inceptionv3():    
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
        x = inceptionv3.preprocess_input(x)
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
        x = inceptionv3.preprocess_input(x)
        X_val.append(x)
    print 'Total val examples: ' + str(len(y_val))
    print 'No hate examples: ' + str(total_no_hate)
    print 'Hate examples: ' + str(total_hate)  
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    return X_train, y_train, X_val, y_val

def get_test_data_inceptionv3(): 
    print '*****TEST*****'
    filenames = os.listdir(TEST_PATH) 
    shuffle(filenames)
    X_test = []
    y_test = []
    total_hate = 0
    total_no_hate = 0
    for filename in filenames:
        if "no_hate" in filename:
            y_test.append(0)
            total_no_hate += 1
        else:
            y_test.append(1)
            total_hate += 1
        img = image.load_img(TEST_PATH+filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = inceptionv3.preprocess_input(x)
        X_test.append(x)
    print 'Total val examples: ' + str(len(y_test))
    print 'No hate examples: ' + str(total_no_hate)
    print 'Hate examples: ' + str(total_hate)  
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_test, y_test






def calculate_evaluation_metrics(y_gt, y_preds):        
    metric_types =  ['micro', 'macro', 'weighted']
    metric_results = {
        'precision' : {},
        'recall' : {},
        'f1' : {},
        'acc' : -1.0        
    }
            
    for t in metric_types:
        metric_results['precision'][t] = metrics.precision_score(y_gt, y_preds, average = t)
        metric_results['recall'][t] = metrics.recall_score(y_gt, y_preds, average = t)
        metric_results['f1'][t] = metrics.f1_score(y_gt, y_preds, average = t)
        metric_results['acc'] = metrics.accuracy_score(y_gt, y_preds) 
                
    return metric_results

if __name__ == "__main__":
    print 'Starting'
    print 'Building model...'
    sys.stdout.flush()
    model = build_xception_model()
    model.summary()
    print 'Building data...'
    X_train, y_train, X_val, y_val = get_training_data_xception()
    sys.stdout.flush()
    print 'X_train: ' + str(len(X_train))
    print(X_train.shape)
    print 'y_train: ' + str(len(y_train))
    print(y_train.shape)
    print 'X_val: ' + str(len(X_val))
    print(X_val.shape)
    print 'y_val: ' + str(len(y_val))
    print(y_val.shape)
    print 'Training mode...'
    sys.stdout.flush()
    results = model.fit(X_train, y_train, epochs= 100, batch_size = 32, validation_data = (X_val, y_val))
    print 'Evaluating last model...'
    sys.stdout.flush()
    X_test, y_test = get_test_data_xception()
    yp = model.predict(X_test, batch_size=32, verbose=1)
    y_preds = np.argmax(yp, axis=1)
    print '*'*30
    print 'PREDS'
    sys.stdout.flush()
    print y_preds
    sys.stdout.flush()
    print '*'*30
    print 'TEST'
    sys.stdout.flush()
    print y_test
    sys.stdout.flush()
    evaluation = calculate_evaluation_metrics(y_test, y_preds)
    print 'accuracy: ', evaluation['acc']
    print 'precision:', evaluation['precision']
    print 'recall:', evaluation['recall']
    print 'f1:', evaluation['f1'] 
    print 'fin'
