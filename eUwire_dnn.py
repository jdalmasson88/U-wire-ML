import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import UAnaTools as utls
import os

plt.ion()

###### READ IGOR's DATASET ######
#test_data,train_data = utls.readIgor('/global/cscratch1/sd/igor_ost/Mike/')

num_test_wfm = 100    ###### number of test waveforms #####
ch = raw_input('select a channel: ')
svfile = 'learning_curve.png'

###### READ JACOPO'S DATASET ######
test_data,train_data = utls.readBoolMike('/global/cscratch1/sd/jdalmass/ascii/ch'+ch+'/QRNWFs_0.dat', num_test_wfm)

###### PREPPING THE DATA (IMAGES AND LABELS, BASELINE SUBTRACTION, NORMALIZATION) ######
x_train,y_train = utls.sepXY(train_data)
x_train_norm,train_min,train_max = utls.NormalizeData(x_train,True)
data_train_norm = utls.DataSet(x_train_norm,y_train)

x_test,y_test = utls.sepXY(test_data)
x_test_norm = utls.NormalizeData(x_test,False,True,train_min,train_max)
data_test_norm = utls.DataSet(x_test_norm,y_test)

###### DESIGNING THE ARCHITECTURE ######
x = tf.placeholder(tf.float32,[None,1000])
y_ = tf.placeholder(tf.float32,[None,1])

W0 = tf.get_variable("W0",shape=[1000,500],initializer=tf.contrib.layers.xavier_initializer())
b0 = utls.bias_var([500])
h0 = tf.nn.relu(tf.matmul(x,W0)+b0)

keep_prob = tf.placeholder(tf.float32)
h0_drop = tf.nn.dropout(h0,keep_prob)

W1 = tf.get_variable("W1",shape=[500,200],initializer=tf.contrib.layers.xavier_initializer())
b1 = utls.bias_var([200])
h1 = tf.nn.relu(tf.matmul(h0_drop,W1)+b1)

W2 = tf.get_variable("W2",shape=[200,100],initializer=tf.contrib.layers.xavier_initializer())
b2 = utls.bias_var([100])
h2 = tf.nn.relu(tf.matmul(h1,W2)+b2)

W3 = tf.get_variable("W3",shape=[100,40],initializer=tf.contrib.layers.xavier_initializer())
b3 = utls.bias_var([40])
h3 = tf.nn.relu(tf.matmul(h2,W3)+b3)

W4 = tf.get_variable("W4",shape=[40,10],initializer=tf.contrib.layers.xavier_initializer())
b4 = utls.bias_var([10])
h4 = tf.nn.relu(tf.matmul(h3,W4)+b4)

W5 = tf.get_variable("W5",shape=[10,2],initializer=tf.contrib.layers.xavier_initializer())
b5 = utls.bias_var([2])
h5 = tf.nn.relu(tf.matmul(h4,W5)+b5)

W_last = tf.get_variable("W_last",shape=[2,1],initializer=tf.contrib.layers.xavier_initializer())
b_last = utls.bias_var([1])
y = tf.nn.relu(tf.matmul(h5,W_last)+b_last)

mse = tf.reduce_mean(tf.square(y-y_))

regularizers = (tf.nn.l2_loss(W0)+tf.nn.l2_loss(b0)+tf.nn.l2_loss(W1)+tf.nn.l2_loss(b1)+
tf.nn.l2_loss(W2)+tf.nn.l2_loss(b2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(b3)+
tf.nn.l2_loss(W4)+tf.nn.l2_loss(b4)+tf.nn.l2_loss(W5)+tf.nn.l2_loss(b5)+
tf.nn.l2_loss(W_last)+tf.nn.l2_loss(b_last))

loss = mse + regularizers*1.e-6

###### OPTIMIZATION AND BACKPROPAGATION ######
train_step = tf.train.AdamOptimizer().minimize(loss)


sess =tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

###### IF param.ckpt EXISTS, THIS STEP RESTORES THE PARAMETERS ######
saver = tf.train.Saver()
saver.restore(sess,'/global/homes/j/jdalmass/uwireML/param.ckpt')  # comment these two
print("Model restored.")                       # lines if reintializing the parameters

trainloss = []
testloss = []

###### RUNNING THE TRAINING/TESTING PART ######
for i in range(10000):
	###### BATCHES PRODUCTION ######
	batch = data_train_norm.next_batch(5000)
	if i%20 == 0:
		ce = mse.eval(feed_dict={x:data_train_norm.images,y_:data_train_norm.labels,keep_prob:1.0},session=sess)
		trainloss.append(ce)
		tce = mse.eval(feed_dict={x:data_test_norm.images,y_:data_test_norm.labels,keep_prob:1.0},session=sess)
		testloss.append(tce)
		print("step %d, test MSE %g"%(i,tce))
	train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.8},session=sess)

trainlossnp = np.asarray(trainloss)
testlossnp = np.asarray(testloss)
pred = y.eval(feed_dict={x:data_test_norm.images,y_:data_test_norm.labels,keep_prob:1.0},session=sess)

print ('final plot MSE: %f'%np.mean(np.square(pred-data_test_norm.labels)))

###### THIS STEP SAVES THE PARAMETERS ######
#save_path = saver.save(sess, "/global/homes/j/jdalmass/uwireML/param.ckpt") #comment these
#print("Model saved in file: %s" % save_path)        #two lines if restoring the parameters

###### PLOTTING LEARNING CURVE AND SCATTER PLOT ######
plt.figure(1)
plt.plot(trainloss, label='train data set')
plt.plot(testloss, label='test data set')
plt.xlabel('batch cycle')
plt.yscale('log')
plt.ylabel('MSE')
plt.legend()
#if os.path.isfile(svfile): os.remove(svfile) 
#plt.savefig('/global/homes/j/jdalmass/uwireML/'+svfile)

plt.figure(2)
plt.scatter(pred,data_test_norm.labels)
plt.xlabel('Reconstructed Energy [keV]')
plt.ylabel('True Energy [keV]')
#if os.path.isfile('/global/homes/j/jdalmass/uwireML/trueVSest.png'): os.remove('/global/homes/j/jdalmass/uwireML/trueVSest.png')
#plt.savefig('/global/homes/j/jdalmass/uwireML/trueVSest.png')
raw_input()
