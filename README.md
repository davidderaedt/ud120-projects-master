# Udacity Introduction to Machine Learning class

This repo stores the code for Udacity's  Introduction to Machine Learning course.

https://classroom.udacity.com/courses/ud120

The course is a great introduction to the fundamentals of Machine Learning and most common tools, concepts and algorithms, based on the scikit-learn library.

The final project (POI identification from the Enron dataset) can be found ran using poi_id.py and tested using tester.py in the final_project folder.

Note that most text learning scrips won't work as is, as they expect a (1Gb+) maildir directory at the upper level, which contains all Enron emails released to the public, and which can be created using the instructions described in the course introduction.

## Final project : Indentification of Person Of Interest using the Enron dataset

The code is quite heavily commented, but as an overview, the process I used was as follows:

* Feature selection, first based on intuition (assuming a correlation between POI status and financial data + email communication with other POIs)
* Data driven evaluation of hypothesis, starting with using classifiers' feature_importances_ attribute
* Visualization using matplotlib (also, I used print a lot)
* Removal of obvious outliers
* Addition of a new, synthetic feature, poi_mail_ratio, which represents the share of emails sent from POI to this person.
* Testing of several classifiers (Gaussian Naive Bayes, Decision Tree, ...) and parameters

I got the best results with a DecisionTree with a min_samples_split of 5, and a training / test data split of 0.3.

## Evaluation (9000 predictions)
Accuracy: 0.82511       
Precision: 0.48159      
Recall: 0.42033 F1: 0.44888     
F2: 0.43130
