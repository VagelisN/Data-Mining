# -*- coding: utf-8 -*-
from sklearn.decomposition import TruncatedSVD
from string import punctuation
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import preprocessing
from gensim.parsing.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

#stopwords with some added words
stopwords = set()

my_words = my_words = [u"said",u"say",u"says",u"just",u"did",u"was",u"were",u"year",u"years",u"like",u"people"]

stopwords=stopwords.union(my_words)


for word in ENGLISH_STOP_WORDS:
    word = unicode(word)
    stopwords.add(word)

print stopwords
#function that removes punctuation from a string
def strip_punctuation(s):
    return ''.join(c for c in s if (c not in punctuation and c != unicode('“', encoding='utf-8') and 
                                    c != unicode('”', encoding='utf-8') and 
                                    c != unicode('’', encoding='utf-8') ))


train_data = pd.read_csv('./datasets/train_set.csv',sep="\t",encoding='utf-8')
test_data = pd.read_csv('./datasets/test_set.csv', sep="\t",encoding='utf-8')

print "My method (SVC with OnevsRestClassifier)..."


print "Starting preprocessing...."

#merge the content and the title
train_data["Content"]=(train_data["Title"])+" "+(train_data["Content"])
test_data["Content"] =(test_data["Title"])+" "+(test_data["Content"])

#remove punctuation and stopwords 
for i in range(0,len(train_data["Content"])):
    temp_doc=u""
    for word in train_data["Content"][i].split():
        word=unicode(word)
        word=unicode(word.lower())
        word=unicode(strip_punctuation(word))
        if unicode(word) not in stopwords and unicode(word) != u"–" and unicode(word) != u"…":
            temp_doc=temp_doc+word+" "
    train_data["Content"].replace(to_replace=train_data["Content"][i],value=temp_doc,inplace=True)

print "Finished removing punctuation and stopwords from train_data"

#remove punctuation and stopwords
for i in range(0,len(test_data["Content"])):
    temp_doc=u""
    for word in test_data["Content"][i].split():
        word=unicode(word)
        word=unicode(word.lower())
        word=unicode(strip_punctuation(word))
        if unicode(word) not in stopwords and unicode(word) != u"–" and unicode(word) != u"…":
            temp_doc=temp_doc+word+" "
    test_data["Content"].replace(to_replace=test_data["Content"][i],value=temp_doc,inplace=True)

print "Finished removing punctuation and stopwords from test_data"

#Stemming
train_data["Content"] = PorterStemmer().stem_documents(train_data["Content"])
print "Finished stemming train_data"

test_data["Content"] = PorterStemmer().stem_documents(test_data["Content"])
print "Finished stemming test_data"


#Transform Category from strings to numbers from 0-4
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])


pipeline = Pipeline([ ('vect', CountVectorizer(ngram_range = (1,2))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', OneVsRestClassifier(svm.SVC(kernel = 'rbf', C = 10000, gamma = 0.0001)))])

X_train = train_data["Content"]

#train the classifier with the whole train_data
pipeline.fit(X_train, y)

print "Finished preprocessing"


X_test=test_data['Content']

#predict the test_data
ypred = pipeline.predict(X_test)

#inverse transform numbers from 0-4 to Category strings
ypred=le.inverse_transform(ypred)

#create the .csv file
raw_data = {
    'ID': test_data['Id'], 
    'Category': ypred}
df = pd.DataFrame(raw_data, columns = ['ID', 'Category'])

df.to_csv('testSet_categories.csv',sep='\t',index=False)