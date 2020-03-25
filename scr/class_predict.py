import pandas as pd
import preprocessing as pr
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


def get_predict(votingC):
    """Defines the class of the customer card"""
    
    big_data_rfm= pr.get_RFM_data()
    big_data_rfm = pr.remove_learned_cart(big_data_rfm)
    big_data_rfm=big_data_rfm.set_index('cart')
    big_data_rfm.index.name = None
    features = ['R','F','M','avg_rec_quant','shop_count','avg_positions_num']
    X = big_data_rfm[features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print('start defines the class of the customer card')
    predictions =(votingC.predict_proba(X)[:,1] >= 0.8).astype(bool) # задается порог определения класса
    big_data_rfm['status'] = predictions
    count = list(predictions).count(False)
    print("____________________")
    print('Сlassification result with threshold set')
    print('The count of False is:', count)
    count_true = list(predictions).count(True)
    print('The count of True is:', count_true)
    print("____________________")
    return big_data_rfm

