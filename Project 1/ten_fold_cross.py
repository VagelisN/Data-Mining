import svm as svm
import random_forest as rf
import nb as nb
import my_method_10fold as my_method
import knn_algorithm as knn
import pandas as pd



raw_data = {
    'Statistic Measure': ["Accuracy","Precision","Recall","F_Measure"], 
    'Naive Bayes': nb.nb_scores,
    'Random Forests': rf.rf_scores,
    'SVM': svm.svm_scores,
    'Knn': knn.knn_scores,
    'My method': my_method.my_method_scores
    }


df = pd.DataFrame(raw_data, columns = ['Statistic Measure','Naive Bayes' ,'Random Forests','SVM','Knn','My method'])
df.to_csv('EvaluationMetric_10fold.csv',sep='\t',index=False)
