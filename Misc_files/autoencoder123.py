from __future__ import division, print_function, absolute_import

import timeit
import numpy as np
import tflearn
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from random import randint



def test():
	#timer 
	#start = timeit.default_timer()
	#Load Dataset
    testSet = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/KDD_Test_41.csv')
    testLabels = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_TestLabels_mat5.csv')
    trainingSet = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/KDD_Train_41.csv')
    trainingLabels = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat5.csv')
    validLabels = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_ValidLabels_int3.csv')
    validSet = pd.read_csv('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/KDD_Valid_41.csv')

    # for acurracy score calculation of different classes we will use its one hot encoded version
    #class2_one_hot_encoded = pd.read_csv('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat2.csv')
    #lass3_one_hot_encoded = pd.read_csv('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat3.csv')
    #class4_one_hot_encoded = pd.read_csv('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat4.csv')

    #Different classes of Test Labels (one_hot_encoding=Flase)
    CL5 = pd.read_csv('C:/Users/Jay/Desktop/dataset/class_kdd_train/class5.csv')
    CL2 = pd.read_csv('C:/Users/Jay/Desktop/dataset/class_kdd_train/class2.csv')
    CL3 = pd.read_csv('C:/Users/Jay/Desktop/dataset/class_kdd_train/class3.csv')
    CL4 = pd.read_csv('C:/Users/Jay/Desktop/dataset/class_kdd_train/class4.csv')



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
    #k = class2_one_hot_encoded.values
    #l = class3_one_hot_encoded.values
    #m = class4_one_hot_encoded.values

    #For accuracy score of classes
    #acc_class2 = np.float32(k) 
    #acc_class2 = np.float32(l)
    #acc_class2 = np.float32(m)


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



    f = randint(0,20)

    # Placeholders to hold data before feeding it into network
    #x = tf.placeholder("float",[None, 41]) 	#for images with shape of None,
    #y = tf.placeholder("float",[None, 5])		#for lables with shape of None,10
    #z = tflearn.layers.core.one_hot_encoding(test_labels_set, n_classes = 5, name = 'one_hot_encoded_testlables')

    # Building the encoder
    encoder = tflearn.input_data(shape=[None, 41])
    encoder = tflearn.fully_connected(encoder, 41,activation='relu')
    encoder = tflearn.fully_connected(encoder, 30,activation='relu')
    encoder = tflearn.fully_connected(encoder, 20,activation='relu')
    encoder = tflearn.fully_connected(encoder, 10,activation='relu')
    encoder = tflearn.fully_connected(encoder, 5, activation='softmax')

    #Stochastic Gradient Descent Optimizer
    #optimizerSGD = tflearn.optimizers.Momentum(learning_rate = 0.001, lr_decay=0.096, decay_step=100,momentum=0.09,name='gradient_Decent')


    #For calculating Accuracy at every step of model training 
    acc= tflearn.metrics.Accuracy()

    # Regression, with mean square error (learn about it more here http://tflearn.org/layers/estimator/)
    net = tflearn.regression(encoder, optimizer= 'RMSprop', learning_rate=0.1,
                             loss='mean_square', metric=acc, shuffle_batches=True)


    # Mpdeling the Neural Network (for details http://tflearn.org/models/dnn/)
    model = tflearn.DNN(net, tensorboard_verbose=0)

    #Checking the shapes and ranks before feeding into network to be sure that we dont run into "TYPE ERROR"
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


    # Training the Neural Network (for details http://tflearn.org/models/dnn/)
    model.fit(test_set, test_labels_set, n_epoch=1, validation_set=(valid_set, valid_labels_set),
              run_id="auto_encoder", batch_size=100,show_metric=True, snapshot_epoch=False)

    # Here I evaluate the model with Test Images and Test Lables, calculating the Mean Accuracy of the model.
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

    # Running a session to feed calculate the confusion matrix
    sess = tf.Session()
    # taking the argumented maximum of the predicted probabilities for generating confusion matrix 
    prediction = tf.argmax(y,1)
    # displaying length of predictions and evaluating them in a session 
    with sess.as_default():
    	print ("Length of predicted labels",len(prediction.eval()))
    	predicted_labels = prediction.eval()
    	print("Length of test set labels",len(class5))
    # Again importing the mnist data with one hot as false because we need to know the truepositive and other values for evaluation
    #Images, Lables, testImages, targetLables = mnist.load_data(one_hot=False)

    # Used Sklearn library for evaluation as tensorflows library was not documented properly 
    # Generated the Confusion Matrix 
    confusionMatrix_5 = confusion_matrix(class5, predicted_labels)
    print("\n"+"\t"+"The confusion Matrix class 5 ")
    print ("\n",confusionMatrix_5)


    # Classification_report in Sklearn provide all the necessary scores needed to succesfully evaluate the model. 
    classification_class_5 = classification_report(class5,predicted_labels, digits=4, 
    				target_names =['Normal','DoS','Probe','U2R','R2I'])
    print("\n"+"\t"+"The classification report for all the 5 classes "+"\n")
    print ("\t",classification_class_5)

    classification_class_2 = classification_report(class2,predicted_labels, digits=4, 
                    target_names =['Normal','Attack'])
    print("\n"+"\t"+"The classification report for 2 Classes "+"\n")
    print ("\t",classification_class_2)


    classification_class_3 = classification_report(class3,predicted_labels, digits=4, 
                    target_names =['Normal','DoS','All Other Attacks'])
    print("\n"+"\t"+"The classification report for 2 Classes "+"\n")
    print ("\t",classification_class_3)

    classification_class_4 = classification_report(class4,predicted_labels, digits=4, 
                    target_names =['Normal','DoS','Probe','U2R'])
    print("\n"+"\t"+"The classification report for 2 Classes "+"\n")
    print ("\t",classification_class_4)

    with sess.as_default():
        acc_class2_shape = tf.shape(acc_class2)
        print("Shape of Acc_class2 is :", acc_class2_shape.eval())


    """
    accuracy_class_2 = model.evaluate(train_set,acc_class2)
    print("\n")
    print("\t"+"Mean accuracy of the model for class 2:", accuracy_class_2)

    accuracy_class_3 = model.evaluate(train_set,acc_class3)
    print("\n")
    print("\t"+"Mean accuracy of the model for class 3:", accuracy_class_3)

    accuracy_class_4 = model.evaluate(train_set,acc_class4)
    print("\n")
    print("\t"+"Mean accuracy of the model for class 4:", accuracy_class_4)
    """


   # stop = timeit.default_timer()
   # time = ((stop-start)/60)
    #print ("The total time for this algorithm is ",time,"min")

test()
