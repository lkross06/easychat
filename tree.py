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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import FreqDist

def import_data(): #imports dataset and splits in train-test split
    import pandas as pd
    import json
    with open("data.json") as intent:
        data = json.load(intent)
    df = pd.DataFrame(data['intents'])
    df_patterns = df[['text', 'intent']]
    df_responses = df[['responses', 'intent']]
    return [df, df_patterns, df_responses]



def get_tree():
    t = DecisionTreeClassifier(
        criterion="gini", 
        splitter="best",
        max_depth = 4,
        min_samples_leaf = 100,
        min_samples_split = 1000,
        random_state=1)
    
    df, inq, rep = import_data()
    print(df.head())
    # x = df.drop("intent", axis=1)
    # y = df["intent"]
    
    # xtrain, ytrain, xtest, ytest = train_test_split(x, y, train_size=0.3)
    # t.fit(xtrain, ytrain)

    # print("tree score: ", t.score(xtest, ytest))

get_tree()