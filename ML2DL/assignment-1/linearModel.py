import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
data_root = '.'


pickle_file = os.path.join(data_root, 'notMNIST_small.pickle')
f = open(pickle_file, "r")
data = pickle.load(f)

train_ft = [row.flatten() for row in data['train_dataset']]
test_ft = [row.flatten() for row in data['test_dataset']]


# clf = LogisticRegression()
# clf.fit(train_ft, data['train_labels'])
# preds = clf.predict(test_ft)

# print accuracy_score(data['test_labels'], preds)
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
   
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]



for clf in classifiers:
	print clf
	clf.fit(train_ft, data['train_labels'])
	preds = clf.predict(test_ft)

	print accuracy_score(data['test_labels'], preds)


