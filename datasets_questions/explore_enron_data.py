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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print ("#of people:", len(enron_data))
print "#of features:", len(enron_data["SKILLING JEFFREY K"])

poi_index=0
for person_name in enron_data :
    if enron_data[person_name]["poi"]==1 :
        print person_name+":", (enron_data[person_name]["total_payments"])
        poi_index+=1

print "#of poi:", poi_index
prenticestocks = enron_data["PRENTICE JAMES"]["exercised_stock_options"] + enron_data["PRENTICE JAMES"]["restricted_stock"]
print "Prentice stock", prenticestocks
print "Colwell to pois", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "Skilling ex stock", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
