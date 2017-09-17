import os
from python_speech_features import mfcc, delta, logfbank
import scipy.io.wavfile as wav
from random import shuffle
import numpy


instruments = {'BassClarinet': 1, 'BassTrombone': 2, 'BbClarinet': 3, 'Cello': 4, 'EbClarinet': 5, 'Marimba': 6, 'TenorTrombone': 7, 'Viola': 8, 'Violin': 9, 'Xylophone': 10}

path = 'test_data'

data = []

for filename in os.listdir(path):
	(rate,sig) = wav.read(path +'\\\\'+ filename)
	mfcc_feat = mfcc(sig,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sig,rate)
	data.append(mfcc_feat[0:100,1])


numpy.save('test_data.npy',numpy.asarray(data))
#numpy.save('classification.npy',numpy.asarray(classification))

