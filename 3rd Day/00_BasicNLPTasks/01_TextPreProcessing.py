# -*- coding: utf-8 -*-
""
#--Created on Tue Mar 19 16:32:49 2019

#@author: vsurampu
#--Hands on Exercises on Basic NLP Tasks

import nltk

#---------------------------- Tokenizing Text -----------------------------------------------#

from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "I want to be a certified artificial intelligence professional"

print(sent_tokenize(example_text))

print(word_tokenize(example_text))

for i in word_tokenize(example_text):print(i)

# ------------ bigrams ----------------#

word_data = 'I want to be a certified artificial intelligence professional'
nltk_tokens = nltk.word_tokenize(word_data)
print(list(nltk.bigrams(nltk_tokens)))

# --------------- n-gram --------------#

from nltk.util import ngrams

def word_grams(words, min=1, max=5):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

print(word_grams(nltk_tokens))

# ------------- Removing Stop Words ------------------------ #

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#example_text = "This is an example showing off stop word filtration."
example_text = 'I want to be a certified artificial intelligence professional'

stop_words = set(stopwords.words("english"))

print("List of the Stop words")
print(stop_words)

words = word_tokenize(example_text)

filtered_sentence = []

for w in words:
	if w not in stop_words:
		filtered_sentence.append(w)

print(filtered_sentence)

#------------- Normalization: Stemming and Lemmatization ---------------# 

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
	print(ps.stem(w))

#------------- Stemming - Another Example ---------------- #
new_text = "It is very important to be pythonly while you are pythoning with python.Python name is derived from the pythons"

words=word_tokenize(new_text)

for w in words:
	print(ps.stem(w))

#----------------- Lemmatization Example --------------------#
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer() 
  
print("rocks when lemmatized :", lemmatizer.lemmatize("rocks")) 
print("corpora when lemmatized :", lemmatizer.lemmatize("corpora"))

#ps = PorterStemmer()
print("rocks when Stemmed :", ps.stem("rocks")) 
print("corpora when Stemmed :", ps.stem("corpora"))

# a denotes adjective in "pos" 
print("better :", lemmatizer.lemmatize("better", pos ="a")) 

#---------- Parts of Speech Tagging ----------# 

example_text = "The training is going great and the day is very fine.The code is working and all are happy about it"
token = nltk.word_tokenize(example_text)

nltk.pos_tag(token)

nltk.download('tagsets')

# We can get more details about any POS tag using help funciton of NLTK as follows.
nltk.help.upenn_tagset("PRP$")
nltk.help.upenn_tagset("JJ$")
nltk.help.upenn_tagset("VBG")

#--------------------- Named Entity Recognition using Spacy -------------------------------#
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import pprint
# Run in console  python -m spacy download en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')

print([(X.text, X.label_) for X in doc.ents])

print([(X, X.ent_iob_, X.ent_type_) for X in doc])

sentences = [x for x in doc.ents]

print(sentences)

displacy.serve(nlp(str(sentences)), style='ent')


#--You can view visualization at: http://localhost:5000/---# 
#--displacy.render(nlp(str(sentences)), style='ent') will give HTML Code --#



