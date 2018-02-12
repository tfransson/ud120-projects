#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import cPickle as pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Number of persons:
print "Number of persons: %i" % len(enron_data)

#Number of items:
print "Number of items: %i" % len(enron_data["GLISAN JR BEN F"])

nb_poi = 0
for i in iter(enron_data):
	if enron_data[i]["poi"] == 1:
		nb_poi += 1

print "Number of persons of interest (''POIs''): %i" % nb_poi
#print type(enron_data)
#lens = [len(enron_data[i]) for i in iter(enron_data)]
#print lens