#import libraries
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

#TODO: when we are done, make a list of all the corpora we need to download from nltk so user doesnt have to download all of nltk

class NLP:
    def __init__(self):
        pass

    '''PRE-PROCESSING FUNCTIONS'''

    def getlemmas(self, string): #simplified function
        return self.lemmatize(self.tokenize(string))

    #returns an array of tokenized words (each slot is one word/proper noun phrase/punctuation)
    def tokenize(self, string): #string = string of words
        return nltk.word_tokenize(string)
    
    #gets POS for a word and returns as valid input for lemmatizer
    def getpos(self, word): #word = word in a string
        pos = str(nltk.pos_tag([word])[0][1])
        if pos.startswith('J'):
            return wordnet.ADJ
        elif pos.startswith('V'):
            return wordnet.VERB
        elif pos.startswith('N'):
            return wordnet.NOUN
        elif pos.startswith('R'):
            return wordnet.ADV
        else:          
            return wordnet.NOUN #default for lemmatize = noun
        
        #TODO: finish this function

    #returns list of tokenized words in their root forms based on part of speech
    def lemmatize(self, words): #words = array of tokenized words
        lemmatizer = WordNetLemmatizer()
        rtn = []
        for word in words:
            pos = self.getpos(word)
            rtn.append(lemmatizer.lemmatize(word, pos=pos))
        return rtn

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