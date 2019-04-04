# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:36:07 2019

@author: vsurampu
"""
#--Text Classification using Amazon review dataset--#

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
import pandas

#DATASET PREPARATION
# load the dataset 
data = open('corpus.txt',encoding="utf8").read() 
labels, texts = [], [] 
for i, line in enumerate(data.split("\n")): 
    content = line.split() 
    labels.append(content[0]) 
    texts.append(" ".join(content[1:]))

# create a dataframe using texts and lables 
trainDF = pandas.DataFrame() 
trainDF['text'] = texts 
trainDF['label'] = labels 

# split the dataset into training and validation datasets  
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label']) 

# label encode the target variable  
encoder = preprocessing.LabelEncoder() 
train_y = encoder.fit_transform(train_y) 
valid_y = encoder.fit_transform(valid_y) 

#FEATURE ENGINEERING
#COUNT VECTORS AS FEATURES
# create a count vectorizer object  
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}') 
count_vect.fit(trainDF['text']) 

# transform the training and validation data using count vectorizer object 
xtrain_count =  count_vect.transform(train_x) 
xvalid_count =  count_vect.transform(valid_x)

#TF-IDF VECTORS AS FEATURES
# word level tf-idf 
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000) 
tfidf_vect.fit(trainDF['text']) 
xtrain_tfidf =  tfidf_vect.transform(train_x) 
xvalid_tfidf =  tfidf_vect.transform(valid_x) 

# ngram level tf-idf  
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000) 
tfidf_vect_ngram.fit(trainDF['text']) 
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x) 
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x) 

# characters level tf-idf 
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000) 
tfidf_vect_ngram_chars.fit(trainDF['text']) 
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)  
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

#MODEL BUILDING
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False): 
    # fit the training dataset on the classifier 
    classifier.fit(feature_vector_train, label) 
    # predict the labels on validation dataset 
    predictions = classifier.predict(feature_vector_valid) 
    if is_neural_net: 
        predictions = predictions.argmax(axis=-1) 
    return metrics.accuracy_score(predictions, valid_y) 

#NAIVE BAYES
# Naive Bayes on Count Vectors 

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count) 
print ("NB, Count Vectors: ", accuracy )

# Naive Bayes on Word Level TF IDF Vectors 

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf) 
print ("NB, WordLevel TF-IDF: ", accuracy )

# Naive Bayes on Ngram Level TF IDF Vectors 

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram) 
print ("NB, N-Gram Vectors: ", accuracy )

# Naive Bayes on Character Level TF IDF Vectors 

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars) 
print ("NB, CharLevel Vectors: ", accuracy )

#LINEAR CLASSIFIER
# Linear Classifier on Count Vectors 

accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count) 
print ("LR, Count Vectors: ", accuracy )

# Linear Classifier on Word Level TF IDF Vectors 
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf) 
print ("LR, WordLevel TF-IDF: ", accuracy )

# Linear Classifier on Ngram Level TF IDF Vectors 
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram) 
print ("LR, N-Gram Vectors: ", accuracy )

# Linear Classifier on Character Level TF IDF Vectors 
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars) 
print ("LR, CharLevel Vectors: ", accuracy) 

