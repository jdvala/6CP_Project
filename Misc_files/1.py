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

# First Layer
encoder_layer = tflearn.fully_connected(x,41,activation='relu',bias=True,restore=True)
	#scope='layer1',reuse=None)

loss = tflearn.losses.L2(encoder_layer, wd = 0.001)

#optimizer function which will try and minimize the loss function

sgd = tflearn.optimizers.SGD(learning_rate = 0.01,name ='SGD').get_tensor()

# we will now define training operations we want to perform on the data

train_ops = tflearn.helpers.trainer.TrainOp(loss,sgd, batch_size = 64)
#Now we will define the trainer function which will take the train_op and create a network

#this is our model
trainer = tflearn.helpers.trainer.Trainer(train_ops,tensorboard_verbose=0)
layer1_weights = encoder_layer.W.eval(session=trainer.session)
l1 = tf.shape(layer1_weights)
print(l1.eval(session=trainer.session))
layer1_bias = encoder_layer.b.eval(session=trainer.session)
#we will use fit method to train the network with our data.
trainer.fit(feed_dicts={x:train_set},n_epoch=1, show_metric=False, run_id='firstlayer_pretraining')
print("Saving weights and biases...") 
trainer.save('layer1')

#Layer two pre-traning
encoder_layer = tflearn.fully_connected(encoder_layer,30,bias=True,restore=True)
#scope='layer1',reuse=True)
#Restore the model
trainer.restore('layer1')
#layer2_weights = (encoder_layer2.W).eval(session=trainer.session)
#layer2_bias = (encoder_layer2.b).eval(session=trainer.session)
#print("shape of weights layer 2 :",tf.shape(test_set).eval(session=trainer.session))
trainer.fit(feed_dicts={x:train_set},n_epoch=1, show_metric=False, run_id='secondlayer_pretraining')
#saving the second layers model
print("Saving weights and biases...") 
trainer.save('layer2')


#Layer three pre-traning 
encoder_layer = tflearn.fully_connected(encoder_layer,20,bias=True,restore=True)
	#scope='layer1',reuse=True)
trainer.restore('layer2')
#layer3_weights = (encoder_layer3.W).eval(session=trainer.session)
#layer3_bias = (encoder_layer3.b).eval(session=trainer.session)

trainer.fit(feed_dicts={x:train_set},n_epoch=1, show_metric=False, run_id='Thirdlayer_pretraining')
print("Saving the weights and biases...")
trainer.save('layer3')


#Layer 4
encoder_layer = tflearn.fully_connected(encoder_layer,10,restore=True)
	#scope='layer1',reuse=True)
#resoting the mdoel
trainer.restore('layer3')
layer4_weights = (encoder_layer.W).eval(session=trainer.session) 
layer4_bias = (encoder_layer.b).eval(session=trainer.session)
trainer.fit(feed_dicts={x:train_set},n_epoch=1, show_metric=False, run_id='Fourthlayer_pretraining')
print("Saving the weights and biases...")
trainer.save('layer4')

#Layer 5 pretraining
encoder_layer =tflearn.fully_connected(encoder_layer,5,activation='softmax',restore=True)
	#scope='layer1',reuse=True)
#restoring the model
trainer.restore('layer4')
layer5_weights = (encoder_layer.W).eval(session=trainer.session)
layer5_bias =(encoder_layer.b).eval(session=trainer.session)
trainer.fit(feed_dicts={x:train_set},n_epoch=1, show_metric=False, run_id='Fifthlayer_pretraining')
print("Saving the weights and biases...")
trainer.save('layer5')

#Training the network

acc = tflearn.metrics.Accuracy()
# Regression, with mean square error (learn about it more here http://tflearn.org/layers/estimator/)
net = tflearn.regression(encoder_layer, optimizer='adam', learning_rate=0.01,loss='mean_square', metric=acc, shuffle_batches=True,trainable_vars=None)

# Mpdeling the Neural Network (for details http://tflearn.org/models/dnn/)
model = tflearn.DNN(net, clip_gradients=5.0, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/', checkpoint_path='/tmp/tflearn_ckpt/', best_checkpoint_path='/tmp/tflearn_ckpt/', max_checkpoints=5, best_val_accuracy=0.0)

model.fit(test_set, test_labels_set, n_epoch=1, validation_set=(valid_set, valid_labels_set),run_id="auto_encoder", batch_size=1 ,show_metric=True, snapshot_epoch=False)

evaluation= model.evaluate(train_set,train_labels_set)
print("\n")
print("\t"+"Mean accuracy of the model is :", evaluation)

# Prediction the Lables of the Images that we give to the model just to have a clear picture of Neural Netwok
lables = model.predict_label(train_set)
print("\n")
print("\t"+"The predicted labels are :",lables)

# Predicted probailites 
y = model.predict(train_set)
print("\n")
print("\t"+"\t"+"\t"+"The predicted probabilities are :" )
print("\n")
print (y[f])
