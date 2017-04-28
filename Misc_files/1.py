from __future__ import division, print_function, absolute_import

import timeit
import numpy as np
import tflearn
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sys
import os
from random import randint


os.system('clear')
#timer 
start = timeit.default_timer()
#Placeholder for the trainer

f = randint(0,20)

#dataset
df = pd.read_csv('/Project/incomplete_project/autoencoder/Kdd_Test_41.csv') 			# test set 
er = pd.read_csv('/Project/incomplete_project/autoencoder/NSL_TestLabels_mat5.csv') 	# test labels
ad = pd.read_csv('/Project/incomplete_project/autoencoder/Kdd_Train_41.csv') 			# train set 
qw = pd.read_csv('/Project/incomplete_project/autoencoder/NSL_TrainLabels_mat5.csv') 	# train labels
tr = pd.read_csv('/Project/incomplete_project/autoencoder/Kdd_Valid_41.csv')			# valid set
yu = pd.read_csv('/Project/incomplete_project/autoencoder/NSL_ValidLabels_int3.csv')    # valid labels
rt = pd.read_csv('/Project/incomplete_project/autoencoder/NSL_TrainLabels_int.csv')
a = df.values
b = ad.values
c = qw.values
d = er.values
e = tr.values
f = yu.values
g = rt.values
test_set = np.float32(a)
train_set = np.float32(b)
train_labels_set = np.float32(c)
valid_labels_set = np.float32(d)
valid_set = np.float32(e)
test_labels_set = np.float32(f)
test_set_for_CM =np.float32(g)
x = tf.placeholder("float",[None, 41])
y = tf.placeholder("float",[None, 41])
print("pretraining a network")
# First Layer
encoder_layer = tflearn.fully_connected(x,41,activation='relu',bias=True,restore=True)
encoder_layer = tflearn.fully_connected(x,30,activation='relu',bias=True,restore=True)
encoder_layer = tflearn.fully_connected(x,20,activation='relu',bias=True,restore=True)
encoder_layer = tflearn.fully_connected(x,10,activation='relu',bias=True,restore=True)
encoder_layer = tflearn.fully_connected(x,5,activation='softmax',bias=True,restore=True)

loss = tflearn.losses.L2(encoder_layer, wd = 0.001)

#optimizer function which will try and minimize the loss function

sgd = tflearn.optimizers.SGD(learning_rate = 0.01,name ='SGD').get_tensor()

# we will now define training operations we want to perform on the data

train_ops = tflearn.helpers.trainer.TrainOp(loss,sgd, batch_size = 64)
#Now we will define the trainer function which will take the train_op and create a network

#this is our model
trainer = tflearn.helpers.trainer.Trainer(train_ops,tensorboard_verbose=0)

#we will use fit method to train the network with our data.
trainer.fit(feed_dicts={x:train_set},n_epoch=1, show_metric=False, run_id='firstlayer_pretraining')
print("Saving weights and biases...") 
trainer.save('model_pretraning')



#Now training the model
print("Traing and fine tuning the model...")
trainer1.restore('model_pretraning')
trainer1.fit(feed_dicts={x:train_set,y:train_labels_set},n_epoch=1, show_metric=True, run_id='firstlayer_pretraining')

