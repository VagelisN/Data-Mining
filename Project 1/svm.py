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
from string import punctuation
from gensim.parsing.porter import PorterStemmer
from time import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
import numpy as np

#add some basic stopwords
stopwords = set()
my_words = ["said","say","says"]

stopwords=stopwords.union(my_words)

print "SVC..."


#load the datasets 
train_data = pd.read_csv('./datasets/train_set.csv',sep="\t")
X = train_data["Content"]

#Transform Category from strings to numbers from 0-4
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

# GridSearch for n_components and for the algorithm tuning are commented
"""

# Running gridsearch with optimal n_components from above
pipeline = Pipeline([ ('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('lsi', TruncatedSVD(n_components = 100)),('clf', svm.SVC(kernel = 'rbf',C = 1000, gamma = 0.001)),
])

params_svm = [{'lsi__n_components':[10, 50, 100, 200, 300, 500]}]
print "GridSearchCV for SVM with n_components = 100"
gs_svm=GridSearchCV(estimator=pipeline,param_grid=params_svm,scoring="accuracy",verbose=2,n_jobs=5,cv=10)

gs_svm.fit(X,y)
scores = gs_svm.cv_results_['mean_test_score']
print scores
plt.plot([10, 50, 100, 200, 300, 500], scores)
plt.legend()
plt.xlabel('n_components')
plt.ylabel('Accuracy')
plt.show()

print('Best params: %s' % gs_svm.best_params_)
print('Best training accuracy: %.3f' % gs_svm.best_score_)

"""


#Split the train set by preserving the percentage of samples for each class.
n_folds = 10 
folds = StratifiedKFold(n_splits = n_folds, random_state = 42)
scores = list()

total_accuracy=0
total_recall=0
total_f_score=0
total_precision=0

#Pipeline definition
pipeline = Pipeline([ ('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('lsi', TruncatedSVD(n_components = 300)) ,('clf', svm.SVC(kernel='rbf',C=10000,gamma=0.0001, random_state = 42))])

i=0

#10-fold cross validation
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
print "Finished 10-fold cross validation for SVM.svc()"
print "Total Accuracy: ", (total_accuracy/10)
print "Total Precision: ", (total_precision/10)
print "Total Recall", (total_recall/10)
print "Total F1 score", (total_f_score/10)

svm_scores=[total_accuracy/n_folds,total_precision/n_folds,total_recall/n_folds,total_f_score/n_folds]