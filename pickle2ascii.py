import pickle 
import numpy as np
import os

pickle_path = '/global/cscratch1/sd/jdalmass/pickles/'
ascii_path = '/global/cscratch1/sd/jdalmass/ascii/'
channels = os.listdir(pickle_path)

for index in xrange(len(channels)):
	filename = pickle_path+channels[index]+'/RN_1WF_0.p'
	if os.path.exists(ascii_path+channels[index]):
         print('skip'+channels[index])
         continue
        os.makedirs(ascii_path+channels[index])
	pinput = open(filename,'rb')
	wfs = pickle.load(pinput)
	pinput.close()

	iwf = np.fft.irfft(wfs)

  	fout = open(ascii_path+channels[index]+'/QRNWFs_0.dat',"w")

	fout.write('col ')
	for j in range(1000):
		ch = 600+j
		fout.write('ch_'+str(ch)+' ')
	fout.write('\n')

	num_w = len(wfs)
	num_s = len(wfs[0])
	for i in range(num_w):
		if wfs[i][num_s-6].real <=0. :
			fout.write('0.0')
		else:
			fout.write(str((wfs[i][num_s-6].real)))
		fout.write(" ")
		for j in range(1000):
			fout.write(str(iwf[i][600+j]))
			fout.write(" ")
		fout.write("\n")


	fout.close()
	print filename
