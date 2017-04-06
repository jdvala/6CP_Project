#Deep Belief Network

import tensorflow as tf
import numpy as np
import math

#Constructing RBM 
# Parameters of RBM
class RBM(object):
		#initializing parameters, Method to initialize the parameters of RBM 
	def __init__(self,input_size,output_size):
		self.input_size =  input_size
		self.output_size = output_size	
		self.epochs = 10 
		self.learning_rate = 0.01
		self.batch_size = 100

		# initializing all the weights and biases as matrices of zeros
		self.w = np.zeros([input_size,output_size], np.float32) #weights of zeros 
		self.bias_hidden = np.zeros([output_size], np.float32) #weights of hidden biase 
		self.bias_visible = np.zeros([input_size], np.float32) #weights of visible biase

	#fitting results from weighted visible layer + biases into sigmoid cure 
	def  fit_h_given_v(self, visible, w, bias_hidden):
		sig_h_given_v = tf.nn.sigmoid(tf.matmul(visible, w)+ bias_hidden)
		return sig_h_given_v
	#fitting results from weighted hidden layer + biases into sigmoid curve 
	def fit_v_given_h(self,hidden, w, bias_visible):
		sig_v_given_h = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w))+ bias_visible)
		return sig_v_given_h
		
	def sample_probabilities (self, probabilities):
		prob = tf.nn.relu(tf.sign(probabilities - tf.random_uniform(tf.shape(probabilities))))
		return prob

		# training the RBMS

	def train(self, data):
		# creating placeholders for weights and biases 
		weights = tf.placeholder("float32", [self._input_size, self._output_size])
		biase_of_hidden = tf.placeholder("float32", [self._output_size])
		biase_of_visible = tf.placeholder("float32", [self._input_size])

		#initializing previous weights and biases 
		previous_weights = np.zeros([self._input_size,self._output_size],np.float32)
		previous_hidden_biase = np.zeros([self._output_size],np.float32)
		previous_visible_biase = np.zeros([self._input_size],np.float32)
		
		#initilizing current weights and biases 
		current_weights = np.zeros([self._input_size,self._output_size],np.float32)
		current_hidden_biase = np.zeros([self._output_size],np.float32)
		current_visible_biase = np.zeros([self._input_size],np.float32)

		#creating placeholder for visible layer that is input layer 
		visible_layer_0 = tf.placeholder("float32",[None, self._input_size])

		#initilizing layers with sample probabilites
		hidden_layer_1 = self.sample_probabilities(self.fit_h_given_v(visible_layer_0, weights,biase_of_hidden)) #hidden layer 1
		visible_layer_1 = self.sample_probabilities(self.fit_v_given_h(hidden_layer_1, weights,biase_of_visible)) #visible layer 1
		hidden_layer_2 = self.fit_h_given_v(visible_layer_1, weights,bias_of_hidden) #hidden layer 2

		#creating gradient 
		positive_grad = tf.matmul(tf.transpose(visible_layer_0), hidden_layer_1)
		negative_grad = tf.matmul(tf.transpose(visible_layer_1), hidden_layer_2)

		#updating learning rates of leayers
		update_lr_weights=weights+self.learning_rate*(positive_grad-negative_grad)/tf.to_float(visible_layer_0)[0]
		update_lr_bias_visible=biase_of_visible+self.learning_rate*tf.reduce_mean(visible_layer_0-visible_layer_1,0)
        update_lr_biase_hidden=biase_of_hidden+self.learning_rate*tf.reduce_mean(hidden_layer_1-hidden_layer_2,0)
        # Finding the error rate
        err = tf.reduce_mean(tf.square(visible_layer_0 - visible_layer_1))
        # Training Loop 
        with tf.Session() as sess:
        	sess.run(tf.initialize_all_variables)
        	#For each epoch 
        	for epoch in range(self.epoch)
        	#for each step in batch
        	for start, end in range(0 ,len(X), self.batch_size)
        		batch = X[start,end]
        		#updating weights and biases 
        		current_weights = sess.run(update_lr_weights, feed_dict={visible_layer_0: batch, weights: previous_weights, biase_of_hidden: previous_hidden_biase, biase_of_visible: previous_visible_biase})
                current_hidden_biase = sess.run(update_lr_biase_hidden, feed_dict={visble_layer_0: batch, weights: previous_weights, biase_of_hidden: previous_hidden_biase, biase_of_visible: previous_visible_biase})
                current_visible_biase = sess.run(update_lr_bias_visible, feed_dict={visble_layer_0: batch, weights: previous_weights, biase_of_hidden: previous_hidden_biase, biase_of_visible: previous_visible_biase})
                previous_weights = current_weights
                previous_hidden_biase = current_hidden_biase
                previous_visible_biase = current_visible_biase
                error = sess.run(err, feed_dict={visible_layer_0:X, weights: current_weights, biase_of_visible: current_visible_biase, biase_of_hidden:current_hidden_biase})
               print ("Epoch: %d" %epoch, "Error is %f" error )
               self.w = previous_weights
               self.bias_hidden = previous_hidden_biase
               self.bias_visible = previous_visible_biase

    #output of RBM            
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.bias_hidden)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)



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


RBM_hidden_sizes = [41, 30, 20, 5]  # create 2 layers of RBM with size 400 and 100

# Since we are training, set input as training data
inpX = train_set

# Create list to hold our RBMs
rbm_list = []

# Size of inputs is the number of inputs in the training set(total number of inputs in trainig set)
input_size = inpX.shape[1]

# For each RBM we want to generate
for i, size in enumerate(RBM_hidden_sizes):
   print ('RBM: ', i, ' ', input_size, '->', size)
   rbm_list.append(RBM(input_size, size))
   input_size = size

# For each RBM in our list
for rbm in rbm_list:
  print ('New RBM:')
  # Train a new one
  rbm.train(inpX)
  # Return the output layer
  inpX = rbm.rbm_outpt(inpX)


#Training each RBM in our list 
#pre- training each rbm by calling train() and then passing on this output to the next RBM
for rbm in rbm_list:
	print("New Rbm")
	#training a new rbm
	rbm.train(inpX)
	inpX = rbm.rbm_outpt(inpX)


#constructing a  neural network


class NN(object):
    def __init__(self, sizes, X, Y):
        # Initialize hyperparameters
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate = 0.01
        self._momentum = 0.0
        self._epoches = 10
        self._batchsize = 100
        input_size = X.shape[1]

        # initialization loop
        for size in self._sizes + [Y.shape[1]]:
            # Define upper limit for the uniform distribution range
            max_range = 4 * math.sqrt(6. / (input_size + size))

            # Initialize weights through a random uniform distribution
            self.w_list.append(
                np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float32))

            # Initialize bias as zeroes
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    # load data from rbm
    def load_from_rbms(self, dbn_sizes, rbm_list):
        # Check if expected sizes are correct
        assert len(dbn_sizes) == len(self._sizes)

        for i in range(len(self._sizes)):
            # Check if for each RBN the expected sizes are correct
            assert dbn_sizes[i] == self._sizes[i]

        # If everything is correct, bring over the weights and biases
        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    # Training method
    def train(self):
        # Create placeholders for input, weights, biases, output
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])

        # Define variables and activation functoin
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        # Define the cost function
        cost = tf.reduce_mean(tf.square(_a[-1] - y))

        # Define the training operation (Momentum Optimizer minimizing the Cost function)
        train_op = tf.train.MomentumOptimizer(
            self._learning_rate, self._momentum).minimize(cost)

        # Prediction operation
        predict_op = tf.argmax(_a[-1], 1)

        # Training Loop
        with tf.Session() as sess:
            # Initialize Variables
            sess.run(tf.global_variables_initializer())

            # For each epoch
            for i in range(self._epoches):

                # For each step
                for start, end in zip(
                        range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                    # Run the training operation on the input data
                    sess.run(train_op, feed_dict={
                        _a[0]: self._X[start:end], y: self._Y[start:end]})

                for j in range(len(self._sizes) + 1):
                    # Retrieve weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])

                print( "Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) ==
                                                                                 sess.run(predict_op,
                                                                                          feed_dict={_a[0]: self._X,
                                                                                                     y: self._Y}))))

nNet = NN(RBM_hidden_sizes, train_set, train_labels_set)
nNet.load_from_rbms(RBM_hidden_sizes,rbm_list)
nNet.train()
