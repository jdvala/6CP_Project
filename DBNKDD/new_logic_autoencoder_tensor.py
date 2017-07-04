from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import operator
import os
from sys import exit
os.system('clear')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#Load Dataset
df = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Test_41.csv')             # test set 
er = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_mat5.csv')     # test labels
ad = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Train_41.csv')            # train set 
qw = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat5.csv')    # train labels
tr = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Valid_41.csv')            # valid set
yu = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_ValidLabels_int3.csv')    # valid labels
rt = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_int.csv')
t = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_int.csv')

a = df.values
b = ad.values
c = qw.values
d = er.values
e = tr.values
f = yu.values
g = rt.values
h = t.values
test_set = np.float32(a)
train_set = np.float32(b)
train_labels_set = np.float32(c)
valid_labels_set = np.float32(f)
valid_set = np.float32(e)
test_labels_set = np.float32(d)
test_set_for_CM =np.float32(h)

#Pretraing Parameters
pre_learning_rate = float(input("Please input the Pretraining learning rate : ")) 
pre_training_epochs = int(input("Please input the Pretraining epochs : "))
pre_batch_size = int(input("Please input the Pretraining batch size : "))
display_step = 1


# Pretraining Network Parameters
pre_n_hidden_1 = int(input("Please input the Pretraing network's Hidden layer 1'st Neurons : ")) # 1st layer num features
pre_n_hidden_2 = int(input("Please input the Pretraing network's Hidden layer 2'nd Neurons : "))# 2nd layer num features 
pre_n_hidden_3 = int(input("Please input the Pretraing network's Hidden layer 3'rd Neurons : "))
pre_n_hidden_4 = int(input("Please input the Pretraing network's Hidden layer 4'th Neurons : "))
pre_n_input = 41 
print("\n\n")

# tf Graph input
X = tf.placeholder("float", [None, pre_n_input])
Y = tf.placeholder("float", [None, 5])

weights = {
    'encoder_pre_h1': tf.Variable(tf.random_normal([pre_n_input, pre_n_hidden_1])),
    'encoder_pre_h2': tf.Variable(tf.random_normal([pre_n_hidden_1, pre_n_hidden_2])),
    'encoder_pre_h3': tf.Variable(tf.random_normal([pre_n_hidden_2, pre_n_hidden_3])),
    'encoder_pre_h4': tf.Variable(tf.random_normal([pre_n_hidden_3, pre_n_hidden_4])),
    'decoder_pre_h1': tf.Variable(tf.random_normal([pre_n_hidden_4, pre_n_hidden_3])),
    'decoder_pre_h2': tf.Variable(tf.random_normal([pre_n_hidden_3, pre_n_hidden_2])),
    'decoder_pre_h3': tf.Variable(tf.random_normal([pre_n_hidden_2, pre_n_hidden_1])),
    'decoder_pre_h4': tf.Variable(tf.random_normal([pre_n_hidden_1, pre_n_input])),
}
biases = {
    'encoder_pre_b1': tf.Variable(tf.random_normal([pre_n_hidden_1])),
    'encoder_pre_b2': tf.Variable(tf.random_normal([pre_n_hidden_2])),
    'encoder_pre_b3': tf.Variable(tf.random_normal([pre_n_hidden_3])),
    'encoder_pre_b4': tf.Variable(tf.random_normal([pre_n_hidden_4])),
    'decoder_pre_b1': tf.Variable(tf.random_normal([pre_n_hidden_3])),
    'decoder_pre_b2': tf.Variable(tf.random_normal([pre_n_hidden_2])),
    'decoder_pre_b3': tf.Variable(tf.random_normal([pre_n_hidden_1])),
    'decoder_pre_b4': tf.Variable(tf.random_normal([pre_n_input])),
}
# Building the encoder
def encoder_layer_one(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    return layer_1

def encoder_layer_two(x):
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2'])) 
    return layer_2

def encoder_layer_three(x):
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    return layer_3
def encoder_layer_four(x):

    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))
    return layer_4


# Building the decoder
def decoder_layer_one(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_pre_h1']),
                                   biases['decoder_pre_b1']))
    return layer_1

def decoder_layer_two(x):
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_pre_h2']),
                                   biases['decoder_pre_b2']))
    return layer_2

def decoder_layer_three(x):
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_pre_h3']),
                                   biases['decoder_pre_b3']))
    return layer_3

def decoder_layer_four(x):    
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['decoder_pre_h4']),
                                   biases['decoder_pre_b4']))
    return layer_4

# Construct model for encoders layers
encoder_layer_one_output = encoder_layer_one(X)                                         #1st layer op
encoder_layer_two_output = encoder_layer_two(encoder_layer_one_output)                  #1st o/p -> 2nd i/p
encoder_layer_three_output = encoder_layer_three(encoder_layer_two_output)              #2nd o/p -> 3rd i/p    
encoder_layer_four_final_output = encoder_layer_four(encoder_layer_three_output)        #3rd o/p -> 4th i/p

# Construct model for decoder layers
decoder_layer_one_output = decoder_layer_one(encoder_layer_four_final_output)           #4th encoder o/p -> 1st decoder i/p
decoder_layer_two_output = decoder_layer_two(decoder_layer_one_output)                  #1st o/p -> 2nd i/p
decoder_layer_three_output = decoder_layer_three(decoder_layer_two_output)              #2nd o/p -> 3rd i/p
decoder_layer_four_final_output = decoder_layer_four(decoder_layer_three_output)        #3rd o/p -> 4th i/p


# Prediction
y_pred = decoder_layer_four_final_output
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(pre_learning_rate).minimize(cost)


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(test_set)/pre_batch_size)
    print("Pretraing the model...\n")
    # Training cycle
    for epoch in range(pre_training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: valid_set})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Pretraining Finished!\n\n")
    
#Saving the weights and biases to be used in the fine tuning of the model

weights_encoder_post_1 = weights['encoder_pre_h1']
weights_encoder_post_2 = weights['encoder_pre_h2']
weights_encoder_post_3 = weights['encoder_pre_h3']
weights_encoder_post_4 = weights['encoder_pre_h4']
    
bias_encoder_post_1 = biases['encoder_pre_b1']
bias_encoder_post_2 = biases['encoder_pre_b2']
bias_encoder_post_3 = biases['encoder_pre_b3']
bias_encoder_post_4 = biases['encoder_pre_b4']



#Finetuning Parameters
post_learning_rate = float(input("Please input the Fintuning learning rate : ")) 
post_training_epochs = int(input("Please input the Finetuning epochs : "))
post_batch_size = int(input("Please input the Finetuing batch size : "))
display_step = 1


#Finetuning Network Parameters
post_n_hidden_1 = int(input("Please input the Finetuning network's Hidden layer 1'st Neurons : ")) # 1st layer num features
post_n_hidden_2 = int(input("Please input the Finetuning network's Hidden layer 2'nd Neurons : "))# 2nd layer num features 
post_n_hidden_3 = int(input("Please input the Finetuning network's Hidden layer 3'rd Neurons : "))
post_n_hidden_4 = int(input("Please input the Finetuning network's Hidden layer 4'th Neurons : "))
post_n_input = 41
# Placeholder for the Labels data
Y = tf.placeholder("float", [None,5])

#Building the 

def encoder(x):
	# Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(np.multiply(x, weights_encoder_post_1),
                                   bias_encoder_post_1))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_encoder_post_2),
                                   bias_encoder_post_2))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_encoder_post_3),
                                   bias_encoder_post_3))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights_encoder_post_4),
                                   bias_encoder_post_4))

    return layer_4

# Construct model
encoder_post_op = encoder(X)
# Prediction
y_pred = encoder_post_op
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
optimizer = tf.train.RMSPropOptimizer(pre_learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(post_learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
predict_op = tf.argmax(y_pred,0)
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(test_set)/post_batch_size)
    print("\nFine-tuning the model...\n")
    # Training cycle
    for epoch in range(post_training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: train_set,Y: train_labels_set})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("\nFine-tuning Finished!")
    predicted_labels = sess.run(predict_op, feed_dict ={X:test_set, Y:test_labels_set})
    print("accuracy = ",accuracy.eval(feed_dict={X: test_set, Y: test_labels_set}))
accuracy = accuracy_score(test_set_for_CM, predicted_labels)
b = 100
printaccuracy = operator.mul(accuracy,b)
print("\nThe Accuracy of the model is :", printaccuracy)

