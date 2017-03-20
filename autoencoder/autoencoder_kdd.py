import tensorflow as tf
import numpy
import pandas as pd

df = pd.read_csv('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD Processed/KDD_Test_41.csv')
ad = pd.read_csv('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD Processed/KDD_Train_41.csv')
qw = pd.read_csv('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD Processed/NSL_TrainLabels_mat4.csv')
er = pd.read_csv('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD Processed/NSL_ValidLabels_int2.csv')
tr = pd.read_csv('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD Processed/KDD_Valid_41.csv')
yu = pd.read_csv('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD Processed/NSL_TestLabels_mat5.csv')


a = df.values
b = ad.values
c = qw.values
d = er.values
e = tr.values
f = yu.values
test_set = numpy.float32(a)
train_set = numpy.float32(b)
train_labels_set = numpy.float32(c)
valid_labels_set = numpy.float32(d)
valid_set = numpy.float32(e)
test_labels_set = numpy.float32(f)



sess = tf.Session()
with sess.as_default():
    j = tf.shape(test_set)
    k = tf.rank(test_set)
    print("Shape of test_set is :",j.eval())
    print("Rank of test_set is :",k.eval())

    zx = tf.shape(train_set)
    xz = tf.rank(train_set)
    print("Shape of train_set is :",zx.eval())
    print("Rank of train_set is :",xz.eval())

    jf = tf.shape(train_labels_set)
    kf = tf.rank(train_labels_set)
    print("Shape of train_labels is :",jf.eval())
    print("Rank of train_labels is :",kf.eval())
                 
    ja = tf.shape(valid_labels_set)
    ka = tf.rank(valid_labels_set)
    print("Shape of valid_labels_set is :",ja.eval())
    print("Rank of valid_labels_set is :",ka.eval())

    js = tf.shape(valid_set)
    ks = tf.rank(valid_set)
    print("Shape of valid_set is :",js.eval())
    print("Rank of valid_set is :",ks.eval())

    jd = tf.shape(test_labels_set)
    kd = tf.rank(test_labels_set)
    print("Shape of test_labels_set is :",jd.eval())
    print("Rank of test_labels_set is :",kd.eval())
