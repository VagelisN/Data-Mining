# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from string import punctuation
from gensim.parsing.porter import PorterStemmer
from time import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
import numpy as np


#includes for lemmatization 

#from nltk.corpus import wordnet
#from nltk.stem.wordnet import WordNetLemmatizer
#import nltk



'''def check_money(s):
    for c in s:
        if unicode(c) == unicode('$', encoding='utf-8') or unicode(c) == unicode('£', encoding='utf-8') or unicode(c) == unicode('€', encoding='utf-8'):
            return u"moneyamount"
    return s
'''

#returns if the word is an adjective verb noun or adverb
#used to cut down the tag returned by nltk.pos_tag
'''def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
'''

#lemmatizer = WordNetLemmatizer()
pstem=PorterStemmer()

#add some basic stopwords
stopwords = set()

my_words = my_words = [u"said",u"say",u"says",u"just",u"did",u"was",u"were",u"year",u"years",u"like",u"people"]

stopwords=stopwords.union(my_words)


for word in ENGLISH_STOP_WORDS:
	word = unicode(word)
	stopwords.add(word)


def strip_punctuation(s):
    return ''.join(c for c in s if (c not in punctuation and
                                    c != unicode('“', encoding='utf-8') and 
                                    c != unicode('”', encoding='utf-8') and 
                                    c != unicode('’', encoding='utf-8') ))



print "My method (SVC with OnevsRestClassifier)..."
#load the datasets 
train_data = pd.read_csv('./datasets/train_set.csv',sep="\t",encoding='utf-8')
test_data = pd.read_csv('./datasets/test_set.csv', sep="\t",encoding = 'utf-8')

print "Starting preprocessing..."

#add the title to the content 
train_data["Content"]=(train_data["Title"])+" "+(train_data["Content"])
test_data["Content"]=(test_data["Title"])+" "+(test_data["Content"])

for i in range(0,len(train_data["Content"])):
    temp_doc=u""
    for word in train_data["Content"][i].split():
        word=unicode(word)
        word=unicode(word.lower())
        word=unicode(strip_punctuation(word))
        #word =check_money(word)
        if unicode(word) not in stopwords and unicode(word) != u"–" and unicode(word) != u"…":
            temp_doc = temp_doc+word+" "

    '''doc=[]
    for word in temp_doc.split():
        word=unicode(word)
        doc.append(word)
    tagged = nltk.pos_tag(doc)
        
    temp_doc=u""
    for word in tagged:
        tmp=unicode(word[0])
        temp_doc= temp_doc + lemmatizer.lemmatize(tmp,get_wordnet_pos(word[1]))+" "
    '''

    train_data["Content"].replace(to_replace=train_data["Content"][i],value=temp_doc,inplace=True)



train_data["Content"] = pstem.stem_documents(train_data["Content"])


for i in range(0,len(test_data["Content"])):
    temp_doc=u""
    for word in test_data["Content"][i].split():
        word=unicode(word)
        word=unicode(word.lower())
        word=unicode(strip_punctuation(word))
        #word =check_money(word)
        if unicode(word) not in stopwords and unicode(word) != u"–" and unicode(word) != u"…":
            temp_doc=temp_doc+word+" "

    '''doc=[]
    for word in temp_doc.split():
        word=unicode(word)
        doc.append(word)
    tagged = nltk.pos_tag(doc)
        
    temp_doc=u""
    for word in tagged:
        tmp=unicode(word[0])
        temp_doc= temp_doc + lemmatizer.lemmatize(tmp,get_wordnet_pos(word[1]))+" "
    '''

    test_data["Content"].replace(to_replace=test_data["Content"][i],value=temp_doc,inplace=True)

test_data["Content"] = pstem.stem_documents(test_data["Content"])

X = train_data["Content"]

#Transform Category from strings to numbers from 0-4
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

#Split the train set by preserving the percentage of samples for each class.
n_folds = 10 


folds = StratifiedKFold(n_splits = n_folds, random_state = 42)
scores = list()

print "10-fold cross validation for OnevsRest using SVC:\n\n"

total_accuracy=0
total_recall=0
total_f_score=0
total_precision=0

#Data transformation initializations
pipeline = Pipeline([ ('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', OneVsRestClassifier(svm.SVC(kernel = 'rbf', C = 10000, gamma = 0.0001)))
])

i=0

#10-fold cross validation
for train,test in folds.split(X, y):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    #Train the classifier with the training split
    pipeline.fit(X_train, y_train)

    #Predict the test split
    y_pred = pipeline.predict(X_test)

    #Get scores for this fold 
    scores.append(y_pred)
    total_accuracy += accuracy_score(y[test], y_pred)
    total_recall += recall_score(y[test], y_pred, average='macro')
    total_f_score += f1_score(y[test], y_pred, average='macro')
    total_precision += precision_score(y[test], y_pred, average='macro')
    #print "Fold  #%d :\n" % i
    #print "\tAccuracy:",accuracy_score(y[test], y_pred)
    #print "\tPrecision:",precision_score(y[test], y_pred, average='macro')
    #print "\tRecall:",recall_score(y[test], y_pred, average='macro')
    #print "\tF_score:",f1_score(y[test], y_pred, average='macro')
    #i += 1

print "Total Accuracy: ", (total_accuracy/n_folds)
print "Total Precision: ", (total_precision/n_folds)
print "Total Recall", (total_recall/n_folds)
print "Total F1 score", (total_f_score/n_folds)

my_method_scores=[total_accuracy/n_folds,total_precision/n_folds,total_recall/n_folds,total_f_score/n_folds]