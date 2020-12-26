import json
import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# we use tokenize to split our sentence into words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
    
# we use stemmer method to take "the root" of the word, for example reject "s" in plural words. 
# Stemmers remove morphological affixes from words, leaving only the word stem.
def stem(word):
    return stemmer.stem(word.lower())

# this method is similar to stemmer, but return word with knowledge of the context
def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

# we use bag of words method to convert sentence into vector with value 1 at the position where words appear
def bag_of_words(token_sentence,all_words):
    token_sentence = [lemmatize(word) for word in token_sentence]
    
    bag = np.zeros(len(all_words),dtype=np.float32)
    for index,word in enumerate(all_words):
        if word in token_sentence:
            bag[index]=1.0
            
    return bag