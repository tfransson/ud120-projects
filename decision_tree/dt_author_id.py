#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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

print("Number of features in training dataset: %i" % len(features_train[0]))
print("Number of entries in labels_train: %i" % len(labels_train))


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_split = 40)
t0 = time()
clf.fit(features_train, labels_train)
print("Training time: %.3f s." % round(time()-t0, 3))

t1 = time()
pred = clf.predict(features_test)
print("Prediction time: %.3f s." % round(time()-t1, 3))


from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)

print("The accuracy is %.3f" % acc)
