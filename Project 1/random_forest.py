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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np

#add some basic stopwords
stopwords = set()
my_words = ["said","say","says"]

stopwords=ENGLISH_STOP_WORDS.union(my_words)

print "RandomForests..."
#load the datasets 
train_data = pd.read_csv('./datasets/train_set.csv',sep="\t")


X = train_data["Content"]

#Transform Category from strings to numbers from 0-4
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

#Split the train set by preserving the percentage of samples for each class.
n_folds = 10 
folds = StratifiedKFold(n_splits = n_folds)


#Grid search for the n_components
"""
pipeline = Pipeline([ ('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('lsi', TruncatedSVD(n_components = 100)),('clf', RandomForestClassifier(max_depth=50,min_samples_leaf=2, n_estimators = 600)),
])

params_rf = {
    'clf__max_depth': [5,10,25,50,75,100]
}

print "GridSearchCV for Random Forest with n_components = 100"
gs_svm=GridSearchCV(estimator=pipeline,param_grid=params_rf,scoring="accuracy",verbose=2,n_jobs=6,cv=10)

gs_svm.fit(X,y)
scores = gs_svm.cv_results_['mean_test_score']
print('Best params: %s' % gs_svm.best_params_)
print('Best training accuracy: %.3f' % gs_svm.best_score_)

plt.plot([5,10,25,50,75,100], scores)
plt.legend()
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.show()
"""

scores = list()
total_accuracy=0
total_recall=0
total_f_score=0
total_precision=0


# Pipeline definition
pipeline = Pipeline([ ('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('lsi', TruncatedSVD(n_components = 200)) ,('clf',RandomForestClassifier(max_depth = 50, min_samples_leaf = 2, n_estimators = 600))
])

i=0
X = train_data["Content"]
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
    #print "\tAccuracy:",accuracy_score(y[test], y_pred)
    #print "\tPrecision:",precision_score(y[test], y_pred, average='macro')
    #print "\tRecall:",recall_score(y[test], y_pred, average='macro')
    #print "\tF_score:",f1_score(y[test], y_pred, average='macro')
    i += 1
print "Finished 10-fold cross validation for RandomForests"
print "Total Accuracy: ", (total_accuracy/n_folds)
print "Total Precision: ", (total_precision/n_folds)
print "Total Recall", (total_recall/n_folds)
print "Total F1 score", (total_f_score/n_folds)

rf_scores=[total_accuracy/n_folds,total_precision/n_folds,total_recall/n_folds,total_f_score/n_folds]