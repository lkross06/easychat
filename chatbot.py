#Lucas Ross and Finn Taylor 19 Apr 2023

#import libraries
import nltk
import random
from nltk.chat.util import Chat, reflections
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

#TODO: when we are done, make a list of all the corpora we need to download from nltk so user doesnt have to download all of nltk

class Chatbot:
    def __init__(self):
        pass

    '''PRE-PROCESSING FUNCTIONS'''

    #returns an array of tokenized words (each slot is one word/proper noun phrase/punctuation)
    def tokenize(self, string): #string = string of words
        return nltk.word_tokenize(string)
    
    #gets POS for a word and returns as valid input for lemmatizer
    def getpos(self, word): #word = word in a string
        pos = nltk.pos_tag(word)
        if str(pos).startswith("N"):
            return "n"

    #returns list of tokenized words in their root forms based on part of speech
    def lemmatize(self, words): #words = array of tokenized words
        lemmatizer = WordNetLemmatizer()
        rtn = []
        for word in words:
            rtn += lemmatizer.lemmatize(word, pos=self.getpos(word))

    #filters through tokenized words and removes "stop words" (i.e. unimportant words like "a" or "the" or "an")
    def filter_stopwords(self, words): #words = array of tokenized words
        #word.casefold() converts to lowercase!!
        return [word for word in words if word.casefold() not in set(stopwords.words("english"))]

    #returns list of name entities (proper nouns) + type
    #list of NE types: organization, person, location, date, time, money, percent, facility, gpe
    def get_ne(self, words): #words = array of tokenized words
        rtn = []
        tags = nltk.pos_tag(words)
        tree = nltk.ne_chunk(tags) #we want specific NE type
        for t in tree: #for each leaf
            if hasattr(t, "label"): #if it has a label (another branch in tree with its NE type)
                for n in t:
                    rtn.append((n[0], t.label())) #n[0] = word, n[1] = pos tag
        return rtn

    '''VISUALIZATION FUNCTIONS'''

    #shows a frequency distribution of words in the text with stop word filter
    def show_freqdist(self, string): #string = raw string of words
        string = self.filter_stopwords(self.tokenize(string)) #remove stop words
        dist = FreqDist(string)
        dist.plot(20, cumulative=True)

    '''ML FUNCTIONS'''

    '''UI FUNCTIONS'''


quote = """
Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
for countless centuries Mars has been the star of war—but failed to
interpret the fluctuating appearances of the markings they mapped so well.
All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
and then by other observers. English readers heard of it first in the
issue of Nature dated August 2.""" #this is just a test quote dont worry abt it

chatbot = Chatbot()
chatbot.show_freqdist(quote)