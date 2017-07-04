
from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
import tensorflow as tf
import pandas as pd
from random import randint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Test_41.csv')             # test set 
er = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_mat5.csv')     # test labels
ad = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Train_41.csv')            # train set 
qw = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat5.csv')    # train labels
tr = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Valid_41.csv')            # valid set
yu = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_ValidLabels_int3.csv')    # valid labels
rt = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_int.csv')
t = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_int.csv')

# Reading classes files for confusion matrics and classification reports
class2_for_test_data = pd.read_csv('/home/jay/Documents/Project/Classes_of_dataset/classes_for_KDD_test_data/class2.csv')
class3_for_test_data = pd.read_csv('/home/jay/Documents/Project/Classes_of_dataset/classes_for_KDD_test_data/class3.csv')
class4_for_test_data = pd.read_csv('/home/jay/Documents/Project/Classes_of_dataset/classes_for_KDD_test_data/class4.csv')


#Taking the values from files for main dataset.
a = df.values
b = ad.values
c = qw.values
d = er.values
e = tr.values
f = yu.values
h = t.values

#Taking the values from csv files for classes datafiles
i = class2_for_test_data.values
j = class3_for_test_data.values
k = class4_for_test_data.values

#Converting them into float values and using it insted
test_set = np.float32(a)
train_set = np.float32(b)
train_labels_set = np.float32(c)
valid_labels_set = np.float32(f)
valid_set = np.float32(e)
test_labels_set = np.float32(d)
test_set_for_CM =np.float32(h)

#Converting the class values into numpy float values
class2 = np.float32(i)
class3 = np.float32(j)
class4 = np.float32(k)

# Random integer to restrict the results 
f = randint(0,20)

# Placeholders to hold data before feeding it into network
x = tf.placeholder("float",[None, 41]) 	#for images with shape of None,784
y = tf.placeholder("float",[None, 5])		#for lables with shape of None,10

# Building the encoder
encoder = tflearn.input_data(shape=[None, 41],activation='relu')
encoder = tflearn.fully_connected(encoder, 30,activation='relu')
encoder = tflearn.fully_connected(encoder, 20,activation='relu')
encoder = tflearn.fully_connected(encoder, 10,activation='relu')
encoder = tflearn.fully_connected(encoder, 5, activation='softmax')

#For calculating Accuracy at every step of model training 
acc= tflearn.metrics.Accuracy()

# Regression, with mean square error (learn about it more here http://tflearn.org/layers/estimator/)
net = tflearn.regression(encoder, optimizer='adam', learning_rate=0.1,
                         loss='mean_square', metric=acc, shuffle_batches=True)


# Mpdeling the Neural Network (for details http://tflearn.org/models/dnn/)
model = tflearn.DNN(net, tensorboard_verbose=0)

# Training the Neural Network (for details http://tflearn.org/models/dnn/)S
model.fit(train_set, train_labels_set, n_epoch=10, validation_set=(valid_set,valid_labels_set),
          run_id="auto_encoder", batch_size=10,show_metric=True, snapshot_epoch=True)

# Here I evaluate the model with Test Images and Test Lables, calculating the Mean Accuracy of the model.
evaluation= model.evaluate(test_set,test_labels_set)
print("\n")
print("\t"+"Mean accuracy of the model is :", evaluation)

# Prediction the Lables of the Images that we give to the model just to have a clear picture of Neural Netwok
lables = model.predict_label(test_labels_set)
print("\n")
print("\t"+"The predicted labels are :",lables)

# Predicted probailites 
y = model.predict(test_labels_set)
print("\n")
print("\t"+"\t"+"\t"+"The predicted probabilities are :" )
print("\n")
print (y[f])

# Running a session to feed calculate the confusion matrix
sess = tf.Session()
# taking the argumented maximum of the predicted probabilities for generating confusion matrix 
prediction = tf.argmax(y,1)
# displaying length of predictions and evaluating them in a session 
with sess.as_default():
	print (len(prediction.eval()))
	predicted_labels = prediction.eval()
# Again importing the mnist data with one hot as false because we need to know the truepositive and other values for evaluation

# Used Sklearn library for evaluation as tensorflows library was not documented properly 
# Generated the Confusion Matrix 
confusionMatrix = confusion_matrix(test_set_for_CM, predicted_labels)
print("\n"+"\t"+"The confusion Matrix is ")
print ("\n",confusionMatrix)

# Classification_report in Sklearn provide all the necessary scores needed to succesfully evaluate the model. 
classification = classification_report(test_set_for_CM,predicted_labels, digits=4, 
				target_names =['class 0','class 1','class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9'])
print("\n"+"\t"+"The classification report is ")

print ("\n",classification)

