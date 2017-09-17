import os
from python_speech_features import mfcc, delta, logfbank
import scipy.io.wavfile as wav
from random import shuffle
import numpy


instruments = {'BassClarinet': 1, 'BassTrombone': 2, 'BbClarinet': 3, 'Cello': 4, 'EbClarinet': 5, 'Marimba': 6, 'TenorTrombone': 7, 'Viola': 8, 'Violin': 9, 'Xylophone': 10}

path = 'yahoo\data'

data = []

classification = []

for filename in os.listdir(path):
	for key, value in instruments.items():
		if key in filename:
			(rate,sig) = wav.read(path +'\\\\'+ filename)
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			fbank_feat = logfbank(sig,rate)
			data.append(mfcc_feat[0:100,1])
			classification.append(value)

list1_shuf = []
list2_shuf = []
index_shuf = list(range(len(data)))
shuffle(index_shuf)
for i in index_shuf:
	list1_shuf.append(data[i])
	list2_shuf.append(classification[i])

numpy.save('train_data.npy',numpy.asarray(list1_shuf[0:211]))
numpy.save('train_class.npy',numpy.asarray(list2_shuf[0:211]))
numpy.save('val_data.npy',numpy.asarray(list1_shuf[211:302]))
numpy.save('val_class.npy',numpy.asarray(list2_shuf[211:302]))



numpy.save('data.npy',numpy.asarray(data))
numpy.save('classification.npy',numpy.asarray(classification))
