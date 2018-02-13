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

"""
	The dictionary keys for each person are: 

	'salary', 'to_messages', 'deferral_payments', 'total_payments', 
	'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
	'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 
	'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'director_fees', 
	'deferred_income', 'long_term_incentive', 'email_address', 'from_poi_to_this_person'

"""

import cPickle as pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


######################    Answer different questions about the dataset:
# Number of persons:
nb_totPersons = len(enron_data)
print "Number of persons: %i" % nb_totPersons

#Number of items:
print "Number of items: %i" % len(enron_data["GLISAN JR BEN F"])


nb_poi = 0
nb_has_email = 0
nb_has_salary = 0
nb_no_totPay = 0
nb_poi_noPay = 0

for i in iter(enron_data):
	if enron_data[i]["poi"] == 1:
		nb_poi += 1
	if enron_data[i]["email_address"] != "NaN":
		nb_has_email += 1
	if enron_data[i]["salary"] != "NaN":
		nb_has_salary += 1
	if enron_data[i]["total_payments"] == "NaN":
		nb_no_totPay += 1
	if enron_data[i]["total_payments"] == "NaN" and enron_data[i]["poi"] == 1: 
		nb_poi_noPay += 1

print "Number of persons of interest (''POIs''): %i" % nb_poi
print "The total value of James Prentice's stock is: %.2f dollars" % enron_data["PRENTICE JAMES"]["total_stock_value"]
print "Wesley Colwell sent %i emails to POIs." % enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "The value of Jeffrey K Skilling's exercised stock options is %.2f" % enron_data["SKILLING JEFFREY K"]["exercised_stock_options"] 
print "Kenneth Lay got %.2f dollars, Jeffrey Skilling got %.2f dollars, and Andrew Fastow got %.2f dollars." \
		%(enron_data["LAY KENNETH L"]["total_payments"], enron_data["SKILLING JEFFREY K"]["total_payments"], enron_data["FASTOW ANDREW S"]["total_payments"])
print "%i persons have no quantified salary." % nb_has_salary
print "%i persons have no email_address." % nb_has_email
#print enron_data
print "No info about the total payments is available for %i persons, or %.1f percent of all persons." % (nb_no_totPay, 100.*nb_no_totPay/nb_totPersons)
print "No info about the total payments is available for %i POIs, or %.1f percent of all POIs." % (nb_poi_noPay, 100.*nb_poi_noPay/nb_poi)


# Import tool to convert dict to array
import sys
sys.path.append('../tools')
from feature_format import featureFormat
from feature_format import targetFeatureSplit

features_list = ["poi", "salary", "total_payments"]
value_list = featureFormat(enron_data, features_list)
label, features = targetFeatureSplit(value_list)
#print label
#print features