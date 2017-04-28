
# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 41 # 1st layer num features
n_hidden_2 = 30 # 2nd layer num features
n_hidden_3 = 20 # 3st layer num features
n_hidden_4 = 10 # 4st layer num features
n_hidden_5 = 5  # 5st layer num features, we will use this for classification.

n_input = 41

#defining a pretraining_network using tensorflow
with tf.Graph().as_default():
	#model variables 
	x = tf.placeholder("float",[None, 41])
	y = tf.placeholder("float",[None, 5])

	weights = {
		'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name="layer_1_weights"),
		'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])name="layer_2_weights"),
		'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])name="layer_3_weights"),
		'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])name="layer_4_weights"),
		'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])name="layer_5_weights"),
	}
	biases = {
		'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])name="layer_1_bias"),
		'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])name="layer_2_bias"),
		'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])name="layer_3_bias"),
		'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])name="layer_4_bias"),
		'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])name="layer_5_bias"),
	}

	#For pretraining we will create a network
	def pretraining_network(x):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1'])) #first layer will use sigmoid activation
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h2']),biases['encoder_b2'])) #second layer will use sigmoid activation
		layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h3']),biases['encoder_b3'])) #third layer will use sigmoid activation
		layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h4']),biases['encoder_b4'])) #fourth layer will use sigmoid activation
		layer_5 = tf.nn.softmax(tf.add(tf.matmul(x, weights['encoder_h5']),biases['encoder_b5'])) #fifth layer will use sigmoid activation
		return layer_5
		
	net = pretraining_network(x)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	
	
	#We will use tflearn trainer_op fucntion to define all the pretrain prarameters for the pretraining
	
	pretrain_op = tflearn.TrainOp(loss=loss, optimizer=optimizer,batch_size=100)
	
	#Now we will create a trainer to train the network
	
	trainer = tflearn.Trainer(train_ops=pretrain_op,tensorboard_verbose=0)
	
	#Now here we will train the model
	
	trainer.fit({x:test_set},n_epoch=1,show_metric=True)
	
	#Invoking tf.saver to save weights and biases
	
	saver = tf.train.Saver()
	save_path = saver.save("/testfiles/dnn.ckpt")
	print("saving the models weights and biases")
	
	
		
