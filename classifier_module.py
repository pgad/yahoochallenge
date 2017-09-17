import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import csv
import os

instruments = {'BassClarinet': 1, 'BassTrombone': 2, 'BbClarinet': 3, 'Cello': 4, 'EbClarinet': 5, 'Marimba': 6, 'TenorTrombone': 7, 'Viola': 8, 'Violin': 9, 'Xylophone': 10}

train_data = np.load('train_data.npy')
train_class = np.load('train_class.npy')
val_data = np.load('val_data.npy')
val_class = np.load('val_class.npy')
test_data = np.load('test_data.npy')

#train_data.tolist()
#train_class.tolist()
#val_data.tolist()
#val_class.tolist()


classifier_log = LogisticRegression(penalty = 'l1', C = 0.10)
classifier_svm = SVC(C=0.10)
classifier_mlp = MLPClassifier()

classifier_log.fit(train_data,train_class)
val_score = classifier_log.score(val_data,val_class)
print(val_score)
predictions = classifier_log.predict(val_data)
print(predictions)

classifier_svm.fit(train_data,train_class)
score = classifier_svm.score(val_data, val_class)
print(score)

classifier_mlp.fit(train_data,train_class)
score = classifier_mlp.score(val_data, val_class)
print(score)

predicitions = classifier_log.predict(test_data)

i = 0

lst = []

for filename in os.listdir('test_data'):
	for key, value in instruments.items():
		if predictions[i] is value:
			lst.append((filename,key))
			i = i + 1

with open('submission.csv', 'w') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(lst)

