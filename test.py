#import libraries
import nltk
import random
import math
import numpy as np
import json
import pandas as pd
from chatbot import Chatbot
from sklearn.tree import DecisionTreeClassifier

'''
THIS FILE IS JUST A TEST TO MAKE SURE ALL OF THE DATASET PARSING IS WORKING
'''

def import_data(): #imports dataset as ["category", "user query", "chatbot response"]
    with open("data.json") as intent: #load jsons
        data = json.load(intent)
    df = pd.DataFrame(data['intents'])
    df2 = df[['intent', "text", "responses"]] #choose specific columns
    df2 = df2.rename(columns={"text":"in", "responses":"out"}) #making it more readable

    return df2

def get_tree():
    t = DecisionTreeClassifier(
        criterion="gini", 
        splitter="best",
        max_depth = 4,
        min_samples_leaf = 100,
        min_samples_split = 1000,
        random_state=1)
    
    df = import_data()
    return df

df = import_data()
df["intent"] = df["intent"].apply(lambda x: str.lower(x))
df["in"] = df["in"].apply(lambda x: [str.lower(n) for n in x])

cb = Chatbot()

#proess "in" and "out" into array of lemmas
def to_lemmas(arr):
    rtn = []
    for i in arr:
        words = cb.lemmatize(cb.tokenize(i))
        words = cb.filter_stopwords(words)
        rtn += words
    return [*set(rtn)]

df["in"] = df["in"].apply(to_lemmas)

arr = df["in"].to_numpy()
arr2 = []
for i in arr:
    arr2 += i
all_keywords = [*set(arr2)] #holds all lemmas for input text
for word in all_keywords:
    df[word] = [0 for x in range(0, df.shape[0])]

for i, row in df.iterrows():
    for word in row["in"]:
        try:
            if word in all_keywords:
                df.loc[i, word] = 1
        except:
            pass

def get_intent(string): #string = user query
    arr = cb.lemmatize(cb.tokenize(string))
    highscore = 0
    intent = ""
    for word in arr:
        for i, row in df.iterrows():
            score = 0
            try:
                if df.loc[i, word] == 1:
                    score += 1
            except:
                pass
            if score > highscore:
                intent = row["intent"]
    if intent != "":
        out = list(df.loc[df["intent"] == intent]["out"]).pop()
        r = random.choice(range(0, len(out)))

        print(">>> " + out[r])
    else:
        print(">>> I don't understand.")

def converse(): #actual user interaction function
        userin = input() #automatically waits for input
        if userin != "quit": #stop if user types "quit"

            get_intent(userin)

            converse() #recurse

print("-" * 40)
print("Speaking with chatbot. Type \"quit\" to end conversation.")
print("-" * 40)
converse()