#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Reduce training sample
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

from sklearn.svm import SVC

#Create classifier
clf = SVC(kernel = "linear")#, C = 1., gamma = 1.)

# Fit classifier and time it 
t0 = time()
clf.fit(features_train, labels_train)
print("Training time: %f seconds." % round(time()-t0, 3))

# Predict and time it
t1 = time()
pred = clf.predict(features_test)
print("Prediction time: %f seconds." % round(time()-t1, 3))

#Accuracy
from sklearn.metrics import accuracy_score
print( accuracy_score(labels_test, pred))


