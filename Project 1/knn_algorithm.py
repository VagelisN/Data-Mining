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
from time import time
import scipy
from sklearn.pipeline import Pipeline
import numpy as np

#add some basic stopwords
stopwords = set()
my_words = ["said","say","says"]
stopwords=ENGLISH_STOP_WORDS.union(stopwords)


#load the datasets 
train_data = pd.read_csv('./datasets/train_set.csv',sep="\t")

#Transform Category from strings to numbers from 0-4
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

#Split the train set by preserving the percentage of samples for each class.
n_folds = 10 
folds = StratifiedKFold(n_splits = n_folds)
scores = list()

#Our implementation of the Knn algorithm

class Knn:
    def __init__(self, k = 5):
        self.k = k
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return

    def find_Knn(self, item):
        distances = []
        for d in self.X_train:
            #distances.append(np.linalg.norm(item-d))
            distances.append(scipy.spatial.distance.minkowski(d, item,p = 2))
            #scipy.spatial.distance.euclidean(d, item)
        #find the K smallest distances
        ind = np.argpartition(distances, self.k)
        clas = [0] * 5
        for i in range(0,self.k):
            if self.y_train[ind[i]] == 0:
                clas[0] +=1
            elif self.y_train[ind[i]] == 1:
                clas[1] +=1
            elif self.y_train[ind[i]] == 2:
                clas[2] +=1
            elif self.y_train[ind[i]] == 3:
                clas[3] +=1
            elif self.y_train[ind[i]] == 4:
                clas[4] +=1
        return clas.index(max(clas)) #return the number of the class
            

    def predict(self, X_test):
        predictions = []
        n = 0
        for item in X_test:
            predictions.append(self.find_Knn(item))
        return predictions

print "10-fold cross validation for K-nearest neighbors:\n\n"

total_accuracy=0
total_recall=0
total_f_score=0
total_precision=0


pipeline = Pipeline([ ('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('lsi', TruncatedSVD(n_components = 150)) ,('clf', Knn())
])


i=0
X = train_data["Content"]


#10-fold cross validation
t1 = time()
for train,test in folds.split(X, y):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    #Train the classifier with the training split
    pipeline.fit(X_train,y_train)

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
print "Finished 10 fold cross validation for Knn"
print "Total Accuracy: ", (total_accuracy/n_folds)
print "Total Precision: ", (total_precision/n_folds)
print "Total Recall", (total_recall/n_folds)
print "Total F1 score", (total_f_score/n_folds)

knn_scores=[total_accuracy/n_folds,total_precision/n_folds,total_recall/n_folds,total_f_score/n_folds]