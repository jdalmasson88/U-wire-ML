import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import UAnaTools as utls

#plt.ion()
in_arr = [i for i in xrange(2,36)]
###### DESIGNING THE ARCHITECTURE ######
x = tf.placeholder(tf.float32,[None,1000])
y_ = tf.placeholder(tf.float32,[None,1])

W0 = tf.get_variable("W0",shape=[1000,500],initializer=tf.contrib.layers.xavier_initializer())
b0 = utls.bias_var([500])
h0 = tf.nn.relu(tf.matmul(x,W0)+b0)

keep_prob = tf.placeholder(tf.float32)
#h0_drop = tf.nn.dropout(h0,keep_prob)

W1 = tf.get_variable("W1",shape=[500,200],initializer=tf.contrib.layers.xavier_initializer())
b1 = utls.bias_var([200])
h1 = tf.nn.relu(tf.matmul(h0,W1)+b1)

W2 = tf.get_variable("W2",shape=[200,100],initializer=tf.contrib.layers.xavier_initializer())
b2 = utls.bias_var([100])
h2 = tf.nn.relu(tf.matmul(h1,W2)+b2)

W3 = tf.get_variable("W3",shape=[100,40],initializer=tf.contrib.layers.xavier_initializer())
b3 = utls.bias_var([40])
h3 = tf.nn.relu(tf.matmul(h2,W3)+b3)
h3_drop = tf.nn.dropout(h3,keep_prob)

W4 = tf.get_variable("W4",shape=[40,10],initializer=tf.contrib.layers.xavier_initializer())
b4 = utls.bias_var([10])
h4 = tf.nn.relu(tf.matmul(h3_drop,W4)+b4)

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

loss = mse + regularizers*1.e-4
saver = tf.train.Saver()
mse_arr=[]
mse_arr1=[]
mse_arr2=[]



num_test_wfm = 1000 
for j,ch in enumerate(in_arr):
	test_data,train_data = utls.readBoolMike('/global/cscratch1/sd/jdalmass/ascii/ch'+str(ch)+'/QRNWFs_0.dat', num_test_wfm)

	x_train,y_train = utls.sepXY(train_data)
	x_train_norm,train_min,train_max = utls.NormalizeData(x_train,True)
	data_train_norm = utls.DataSet(x_train_norm,y_train)

	x_test,y_test = utls.sepXY(test_data)
	x_test_norm = utls.NormalizeData(x_test,False,True,train_min,train_max)
	data_test_norm = utls.DataSet(x_test_norm,y_test)
	
	with tf.Session() as ssn:
		train_step = tf.train.AdamOptimizer(1.e-5).minimize(loss)
		ssn.run(tf.initialize_all_variables())
		for i in range(10000):
			batch = data_train_norm.next_batch(1000)
			train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:.7},session=ssn)
		MSE = mse.eval(feed_dict={x:data_test_norm.images,y_:data_test_norm.labels,keep_prob:1.},session=ssn)
		print ('first MSE: %f'%MSE)
                mse_arr.append(MSE)
		save_path = saver.save(ssn,'/global/cscratch1/sd/jdalmass/parameters/param_ch%i.ckpt'%ch)


        with tf.Session() as ssn1:
                train_step = tf.train.AdamOptimizer(1.e-4).minimize(loss)
                ssn1.run(tf.initialize_all_variables())
		saver.restore(ssn1,'/global/cscratch1/sd/jdalmass/parameters/param_ch%i.ckpt'%ch)
                for i in range(10000):
                        batch = data_train_norm.next_batch(1000)
                        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:.7},session=ssn1)
                MSE = mse.eval(feed_dict={x:data_test_norm.images,y_:data_test_norm.labels,keep_prob:.7},session=ssn1)
                print ('second MSE: %f'%MSE)
                mse_arr1.append(MSE)
                save_path = saver.save(ssn1,'/global/cscratch1/sd/jdalmass/parameters/param_ch%i.ckpt'%ch)

        with tf.Session() as ssn2:
                train_step = tf.train.AdamOptimizer(1.e-6).minimize(loss)
                ssn2.run(tf.initialize_all_variables())
                saver.restore(ssn2,'/global/cscratch1/sd/jdalmass/parameters/param_ch%i.ckpt'%ch)
                for i in range(10000):
                        batch = data_train_norm.next_batch(1000)
                        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.},session=ssn2)
                MSE = mse.eval(feed_dict={x:data_test_norm.images,y_:data_test_norm.labels,keep_prob:1.},session=ssn2)
                print ('final MSE: %f'%MSE)
		mse_arr2.append(MSE)
                save_path = saver.save(ssn2,'/global/cscratch1/sd/jdalmass/parameters/param_ch%i.ckpt'%ch)
		with open('output_MSE.txt','a') as out_file:
			np.savetxt(out_file,(ch,mse_arr[j],mse_arr1[j],mse_arr2[j]))

###### PLOTTING LEARNING CURVE AND SCATTER PLOT ######
plt.figure(1)
plt.scatter(in_arr,mse_arr,label='first stage',color='blue')
plt.scatter(in_arr,mse_arr1,label='second stage',color='red')
plt.scatter(in_arr,mse_arr2,label='third stage',color='green')
plt.xlabel('channel')
plt.yscale('log')
plt.ylabel('MSE')
plt.legend()
plt.savefig('full.png')
plt.figure(2)
plt.scatter(in_arr,mse_arr2)
plt.xlabel('channel')
plt.ylabel('MSE')
plt.savefig('last_inter.png')
