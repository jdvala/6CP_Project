import yadlt.models.boltzman as yt
import tensorflow as tf
import numpy as np
import tflearn
import pandas as pd

testSet = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/KDD_Test_41.csv')
testLabels = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_TestLabels_mat5.csv')
trainingSet = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/KDD_Train_41.csv')
trainingLabels = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat5.csv')
validLabels = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_ValidLabels_int3.csv')
validSet = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/KDD_Valid_41.csv')

# for acurracy score calculation of different classes we will use its one hot encoded version
class2_one_hot_encoded = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat2.csv')
class3_one_hot_encoded = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat3.csv')
class4_one_hot_encoded = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat4.csv')

#Different classes of Test Labels (one_hot_encoding=Flase)
CL5 = pd.read_csv('C:/Users/Jay/Desktop/Project/Classes_of_dataset/class5.csv')
CL2 = pd.read_csv('C:/Users/Jay/Desktop/Project/Classes_of_dataset/class2.csv')
CL3 = pd.read_csv('C:/Users/Jay/Desktop/Project/Classes_of_dataset/class3.csv')
CL4 = pd.read_csv('C:/Users/Jay/Desktop/Project/Classes_of_dataset/class4.csv')



a = testSet.values
b = trainingSet.values
c = trainingLabels.values
d = validLabels.values
e = validSet.values
f = validLabels.values
g = CL5.values
h = CL2.values
i = CL3.values
j = CL4.values
k = class2_one_hot_encoded.values
l = class3_one_hot_encoded.values
m = class4_one_hot_encoded.values

#For accuracy score of classes
acc_class2 = np.float32(k) 
acc_class2 = np.float32(l)
acc_class2 = np.float32(m)


#Training Validation and Test Data
test_set = np.float32(a)
train_set = np.float32(b)
train_labels_set = np.float32(c)
valid_labels_set = np.float32(d)
valid_set = np.float32(e)
test_labels_set = np.float32(f)
# this is for all the 5 classes 
class5 =np.float32(g) 
#2 Classes
class2 =np.float32(h)
# 3 classes
class3 =np.float32(i) 
# 4 Classes
class4 =np.float32(j) 


#DBN
my_DBN = yt.DeepBeliefNetwork(rbm_layers=[30,20,10,5], name='dbn', do_pretrain=False,rbm_num_epochs=[10], rbm_gibbs_k=[1],
							rbm_gauss_visible=False, rbm_stddev=0.1, rbm_batch_size=[10],rbm_learning_rate=[0.01], finetune_dropout=1,
							finetune_loss_func='softmax_cross_entropy',finetune_act_func=tf.nn.sigmoid, finetune_opt='sgd',
							finetune_learning_rate=0.001, finetune_num_epochs=10,finetune_batch_size=20, momentum=0.5)
my_DBN.pretrain(train_set)

my_DBN._train_model(train_set,train_labels_set, valid_set,valid_labels_set)

evaluation = my_DBN.evaluate()
print(evaluation)