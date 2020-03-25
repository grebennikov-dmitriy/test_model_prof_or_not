import pandas as pd
from collections import Counter
import pickle
from tqdm import tqdm
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from tabulate import tabulate
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import scipy.stats as stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

# Переходим к обучению 

class Class_Fit(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self,x):
        return self.clf.predict_proba(x)

    def grid_search(self, parameters, Kfold):
        self.grid = GridSearchCV(estimator = self.clf, param_grid = parameters,  n_jobs=6, cv = Kfold)

    def grid_fit(self, X, Y):
        self.grid.fit(X, Y)

    def grid_predict(self, X, Y):
        self.predictions = self.grid.predict(X)
        
def fit_model(X_train, X_test, Y_train, Y_test, X_2, Y_2, X_3, Y_3):
    """ Learn the classifier, prints metrics"""
    
    # Logistic Regression 
    lr = Class_Fit(clf = linear_model.LogisticRegression)
    lr.grid_search(parameters = [{'C':np.logspace(-2,2,20)}], Kfold = 5)
    lr.grid_fit(X = X_train, Y = Y_train)
    lr.grid_predict(X_test, Y_test)
    # ExtraTreesClassifier
    etc = Class_Fit(clf = ensemble.ExtraTreesClassifier)
    param_grid =  {'criterion': ['gini', 'entropy'],
                           'max_depth': [4, 10, 20],
                           'min_samples_split' : [2, 4, 8],
                           'max_depth' : [3, 10, 20]}
    etc.grid_search(parameters = param_grid, Kfold = 5)
    etc.grid_fit(X = X_train, Y = Y_train)
    etc.grid_predict(X_test, Y_test)    
    #Gradient Boosting Classifier
    gb = Class_Fit(clf = ensemble.GradientBoostingClassifier)
    param_grid = {'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    gb.grid_search(parameters = param_grid, Kfold = 5)
    gb.grid_fit(X = X_train, Y = Y_train)
    gb.grid_predict(X_test, Y_test)
    #AdaBoost Classifier
    ada = Class_Fit(clf = AdaBoostClassifier)
    param_grid = {'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    ada.grid_search(parameters = param_grid, Kfold = 5)
    ada.grid_fit(X = X_train, Y = Y_train)

    ada.grid_predict(X_test, Y_test)
    
    # объединяем
    gb_best  = ensemble.GradientBoostingClassifier(**gb.grid.best_params_)
    lr_best  = linear_model.LogisticRegression(**lr.grid.best_params_)
    ada_best  = AdaBoostClassifier(**ada.grid.best_params_)
    etc_best = ensemble.ExtraTreesClassifier(**etc.grid.best_params_)
    #votingC = ensemble.VotingClassifier(estimators=[('gb', gb_best),('ada', ada_best),
    #                                          ('lr', lr_best),('etc', etc_best)], voting='soft')# когда признаков станет больше, можно будет использовать даныый ансамбль 
    votingC = ensemble.VotingClassifier(estimators=[('gb', gb_best)], voting='soft')
    # и обучаю его
    votingC = votingC.fit(X_train, Y_train)
    predictions_baseline = votingC.predict(X_test) # baseline
    print("____________________")
    print('Baseline model metrics:')                      
    print("Precision: {:.2f} % ".format(100*metrics.precision_score(Y_test, predictions_baseline)))
    print("Accuracy: {:.2f} % ".format(100*metrics.accuracy_score(Y_test, predictions_baseline)))
    print("Recall: {:.2f} % ".format(100*metrics.recall_score(Y_test, predictions_baseline,average='binary')))
    print("F1_score: {:.2f} % ".format(100*metrics.f1_score(Y_test, predictions_baseline, average='micro')))
    print("AUC&ROC: {:.2f} % ".format(100*metrics.roc_auc_score(Y_test, predictions_baseline)))
    print("____________________")
    predictions_3=(votingC.predict_proba(X_2)[:,1] >= 0.8).astype(bool)
    print('Test_2 threshold metrics :')
    print("Precision: {:.2f} % ".format(100*metrics.precision_score(Y_2, predictions_3)))
    print("Accuracy: {:.2f} % ".format(100*metrics.accuracy_score(Y_2, predictions_3)))
    print("Recall: {:.2f} % ".format(100*metrics.recall_score(Y_2, predictions_3,average='binary')))
    print("F1_score: {:.2f} % ".format(100*metrics.f1_score(Y_2, predictions_3, average='micro')))
    print("AUC&ROC: {:.2f} % ".format(100*metrics.roc_auc_score(Y_2, predictions_3)))
    count = list(predictions_3).count(False)
    print('The count of False is:', count)
    count_true = list(predictions_3).count(True)
    print('The count of True is:', count_true)
    print("____________________")
    predictions_4=(votingC.predict_proba(X_3)[:,1] >= 0.8).astype(bool)
    
    print('Only prof threshold metrics:')
    print("Precision: {:.2f} % ".format(100*metrics.precision_score(Y_3, predictions_4)))
    print("Accuracy: {:.2f} % ".format(100*metrics.accuracy_score(Y_3, predictions_4)))
    print("Recall: {:.2f} % ".format(100*metrics.recall_score(Y_3, predictions_4,average='binary')))
    print("F1_score: {:.2f} % ".format(100*metrics.f1_score(Y_3, predictions_4, average='micro')))

    count = list(predictions_4).count(False)
    print('The count of False is:', count)
    count_true = list(predictions_4).count(True)
    print('The count of True is:', count_true)
    print("____________________")
    predictions_5=votingC.predict(X_3)
    print('Only prof metrics with baseline classifier:')
    print("Precision: {:.2f} % ".format(100*metrics.precision_score(Y_3, predictions_5)))
    print("Accuracy: {:.2f} % ".format(100*metrics.accuracy_score(Y_3, predictions_5)))
    print("Recall: {:.2f} % ".format(100*metrics.recall_score(Y_3, predictions_5,average='binary')))
    print("F1_score: {:.2f} % ".format(100*metrics.f1_score(Y_3, predictions_5, average='micro')))

    count = list(predictions_5).count(False)
    print('The count of False is:', count)
    count_true = list(predictions_5).count(True)
    print('The count of True is:', count_true)
    return votingC

def dump_classifier(X_train, X_test, Y_train, Y_test, X_2, Y_2, X_3, Y_3):
    """dump our classifire to pickle"""
    votingC = fit_model(X_train, X_test, Y_train, Y_test, X_2, Y_2, X_3, Y_3)
    with open('data/votingC.pkl', 'wb') as fid:
        pickle.dump(votingC, fid)    
    return votingC