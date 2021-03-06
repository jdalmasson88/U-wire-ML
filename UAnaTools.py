import numpy as np
import tensorflow as tf

def readIgor(path):
        train_arr = np.zeros([0,1001])
        for i in xrange(19):
                arr = np.genfromtxt(path+'onech_dats/e1chRNWFs_'+str(i)+'.dat',delimiter=' ')
                d_arr = np.delete(arr,0,0)
                temparr = d_arr.astype(np.float32,copy=False)
                train_arr = np.append(train_arr,temparr,axis=0)
		print ('file '+str(i)+' loaded')
        t_arr = np.genfromtxt(path+'e1chRNWFs_19.dat',delimiter=' ')
        ttemparr = np.delete(t_arr,0,0)
        test_arr = ttemparr.astype(np.float32,copy=False)
        return (test_arr,train_arr)

def readBoolMike(filename, num_test_sample):
	temp0 = np.genfromtxt(filename,delimiter=' ')
	temp = np.delete(temp0,0,0)
	temparr = temp.astype(np.float32,copy=False)
	print ('loaded')
	sarr = np.split(temparr,[num_test_sample])
        return(sarr[0],sarr[1])

def sepXY(data):
	x_temp = data[:,1:]
        y_temp = data[:,0]
	r_y_temp = y_temp.reshape((len(y_temp),1))
        return(x_temp,r_y_temp)

def NormalizeData(data,maxmin=False,base=True,ymin=-1,ymax=-1):
	if base==True:
		sliced_data = data[:,0:300]
        	avg_baseline = np.mean(sliced_data, axis=1)
        	base_data = data.T-avg_baseline
	else:
		base_data = data.T
	if ymin == -1: ymin = base_data.min()
	if ymax == -1: ymax = base_data.max()
        norm_data = (base_data-ymin)/(ymax-ymin)
	if maxmin==True: return norm_data.T,ymin,ymax
	else: return norm_data.T

##################################
class DataSet(object):
  def __init__(self, images, labels):
    assert images.shape[0] == labels.shape[0], (
      "images.shape: %s labels.shape: %s" % (images.shape,
                                             labels.shape))
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def get_col_errors(pred,true):
	errarr = np.empty((0,),np.float32)
	for i in range(len(pred)):
		if true[i]>0:
			errarr = np.append(errarr,pred[i]-true[i],axis=0)
	return errarr	

def get_feature(session,W,i):
	feature = np.empty((1000,1),np.float32)
	warr = session.run(W)
	sum_wj2 = 0.
	for j in range(len(warr)):
		sum_wj2 = sum_wj2 + (warr[j,i]*warr[j,i])
	for j in range(len(warr)):
		feature[j] = warr[j,i]/np.sqrt(sum_wj2)
	return feature

def weight_var(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_var(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)


def smooth(x, wl=31):
	s = np.r_[2*x[0]-x[wl-1::-1],x,2*x[-1]-x[-1:-wl:-1]]
	w = np.ones(wl,'d')
	y = np.convolve(w/w.sum(),s,mode='same')
	return y[wl:-wl+1]
