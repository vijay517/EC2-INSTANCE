import numpy as np
import pandas as pd   
import tensorflow as tf   
from tensorflow import keras
from time import gmtime, strftime 
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import boto3, re, sys, math, json, os, urllib.request,time
import subprocess
import sys
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
args, _ = parser.parse_known_args()

data_dir = args.data

x_train  = pd.read_csv(os.path.join(data_dir, 'x_train.csv')).iloc[:,1:9]
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).iloc[:,1:]

x_test = pd.read_csv(os.path.join(data_dir, 'x_test.csv')).iloc[:,1:9]
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).iloc[:,1:]

file = open(os.path.join(data_dir, 'model_info.txt'))
modelname = file.read().replace("\n", " ")

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch  
  df = pd.DataFrame({"Epoch":hist['epoch'],"MAE":hist['mean_absolute_error'],"MSE":hist['mean_squared_error']})  
  return df

def build_model():
    model = keras.Sequential([
      layers.Dense(30, activation=tf.nn.relu, input_shape=[len(x_train.keys())]),
      layers.Dense(25, activation=tf.nn.relu),
      layers.Dense(15, activation=tf.nn.relu),
      layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    optimizer = tf.keras.optimizers.SGD(lr=0.01)
    
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

if __name__ =='__main__':
    print("Checkpoint-A")
    model = build_model()
    data_location = "/opt/ml/model/"
    history = model.fit(x_train,y_train,epochs=7000,verbose=0,validation_split = 0.2)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch    
    print("Checkpoint-B")
    error = plot_history(history)
    test_predictions = model.predict(x_test).flatten()
    
    pred = pd.DataFrame({"predictions":test_predictions,"ytest":y_test.iloc[:,0]})

    score = sm.r2_score(pred['predictions'].to_numpy().flatten(), pred['ytest'].to_numpy().flatten())
    with open("model_info.txt","a") as file:
      file.write(modelname+"\n"+str(score))
    #text_file = open("accuracy.txt", "wt")
    #n = text_file.write(str(score))
    #text_file.close()

    pred.to_csv("pred.csv")
    error.to_csv("error.csv")
    
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file("pred.csv", 'fypcementbucket','models/{}/pred.csv'.format(modelname))
    s3.meta.client.upload_file("error.csv", 'fypcementbucket','models/{}/error.csv'.format(modelname))
    s3.meta.client.upload_file("model_info.txt", 'fypcementbucket','models/{}/model_info.txt'.format(modelname))   
    tf.contrib.saved_model.save_keras_model(model, data_location)