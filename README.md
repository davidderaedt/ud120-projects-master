# Udacity Introduction to Machine Learning class

This repo stores the code for Udacity's *Introduction to Machine Learning* course.

https://classroom.udacity.com/courses/ud120

The course is a great introduction to the fundamentals of Machine Learning and most common tools, concepts and algorithms, based on the *scikit-learn* library.

The final project (POI identification from the Enron dataset) can be found ran using *poi_id.py* and tested using *tester.py* in the final_project folder.

Note that most text learning scrips won't work as is, as they expect a (1Gb+) *maildir* directory in the parent directory, which contains all Enron emails released to the public, and which can be created using the instructions described in the course introduction.

## Final project : Identification of Person Of Interest using the Enron dataset

### Context
First, it's critical to get a sense of the context here. To understand what we're trying to do, it's important to know about the Enron scandal. I'd suggest the corresponding wikipedia entry:
https://en.wikipedia.org/wiki/Enron_scandal

An interesting side effect of this otherwise sad story is that tons of data (including financial data and emails) were released to the public, which is a great source for data science project.

The goal of this exercise is to identify people considered as "person of interest", ie people who may have played a role of any kind in this gigantic fraud, out of this data.

And since we're talking about real-world data, it's imperfect by definition. It's too short (146 entries), many data points are missing, but it's also what makes it a good candidate for an evaluation (also the data was partly pre-processed for the sake of the exercise).

### Methodology

We're talking about building a binary classifier, ie we'll put people in either of 2 categories: POI or non POI.

The code is quite heavily commented, but as an overview, the process I used was as follows:

1. Look at the data.
2. Make hypothesis.
3. Test hypothesis.

More precisely:

* Looking at the data. Although it may seem obvious, it's a critical step to carefully analyze the dataset at a macro level (e.g. how many entries we have) and micro level (printing random entries to say how it looks)
* Feature selection, first based on intuition. Here I assumed a correlation between POI status and financial data + email communication with other POIs.
* Data driven evaluation of my hypothesis, starting with using classifiers' feature_importances_ attribute. I tested all of them, and ended up ditching most to just keep four : *deferred_income*, *total_stock_value*, *expenses*, and my own *poi_mail_ratio* (more about this below)
* Visualisation using *matplotlib*. Also, I tend to use *print* a lot.
* Removal of obvious outliers. Here I took quite a conservative approach, strictly removing what seemed absolutely necessary (2 non-person entries), as I did not want to remove potentially useful information.
* At least considering data normalization, even if I eventually used a type of classifier which makes this unnecessary (Decision Tree).
* Addition of a new, synthetic feature, *poi_mail_ratio*, which represents the share of emails sent from POI to this person.
* Testing of several classifiers (Gaussian Naive Bayes, Decision Tree, ...) and parameters


## Evaluation (9000 predictions)

I got the best results with a DecisionTree with a min_samples_split of 5, and a training / test data split of 0.3.

Accuracy: 0.82511       
Precision: 0.48159      
Recall: 0.42033
F1: 0.44888     
F2: 0.43130


## Further thoughts

I know for a fact that those results can be improved using better feature selection and parameters for the model.

But I'm also convinced that only email content analysis could improve those results dramatically. The data set is very large, and contains a huge amount of useful data to better classify Enron employees as POI.

Text vectorization was a very interesting first step in that direction, but at this stage, I'd be curious to see how an RNN such as an LSTM could be applied here.
