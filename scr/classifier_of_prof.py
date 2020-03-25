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




def get_RFM_data():
    """ load rfm report"""
    server = 'gp.data.lmru.tech'
    database = 'adb'
    username = '60075437'
    password = 'Passwd321'
    table = 'cards_tic_agg_marts.v_weekly_purchases_by_dept'
    engine = create_engine('postgres://{}:{}@{}/{}'.format(username, password,  server, database))
    sql ="""
    SELECT cod_cartfid ,
           current_date -max(monday_of_week) as R,
           count(DISTINCT monday_of_week) as F,
           SUM(revenue) AS M,
           COUNT(DISTINCT num_ett) AS shop_count,
           avg(positions_num) as avg_positions_num
           
    FROM cards_tic_agg_marts.v_weekly_purchases_by_dept

    group by cod_cartfid    """.format(table)
    big_data_rfm = pd.read_sql(sql, engine)
    big_data_rfm =big_data_rfm.rename(columns={'cod_cartfid':'cart', 'r':'R','m':'M', 'f':'F'})
    # Загрузим данные по чекам
    with open('data/avg_rec_per_user_part1.pkl', 'rb') as fid:
        avg_rec_per_user_part1 = pickle.load(fid)
    with open('data/avg_rec_per_user_part2.pkl', 'rb') as fid:
        avg_rec_per_user_part2 = pickle.load(fid)
    avg_rec_per_user= pd.concat([avg_rec_per_user_part1,avg_rec_per_user_part2])
    big_data_rfm= big_data_rfm.merge(avg_rec_per_user, on = 'cart')
    print(big_data_rfm.head())
    return big_data_rfm
 
# Оставим карты с 17-ю цифрами 
def get_cards_17_digits(big_data_rfm):
    """ get cards with 17 digits"""
    big_data_rfm.loc[:,'len_cart'] = big_data_rfm.cart.apply(lambda x: len(x))
    big_data_rfm=big_data_rfm[big_data_rfm.len_cart==17]
    return big_data_rfm

def assign_prof_cards(cart):
    """True if the 8th digit in the card is 3, otherwise False """
    list1 = [cart]
    result =[j for i in list1 for j in i]
    key_number =result[7]
    if key_number=='3':
        x=True
    else:
        x=False
    return x

def get_column_prof_flag(big_data_rfm):
    
    big_data_rfm.loc[:,'prof_not'] =big_data_rfm.cart.apply(assign_prof_cards)
    del big_data_rfm['len_cart']
    return  big_data_rfm

def get_balance_dataset(big_data_rfm):
    """Сreates a class-balanced dataset"""
    only_prof =big_data_rfm[big_data_rfm.prof_not==1]
    quantile_10percent = only_prof.M.quantile(0.1)
    only_prof =only_prof[only_prof['M']>quantile_10percent]   # убираем 10% самых низких по Monetary проф-карт, это увеличит F1_score примерно на 9%
    n=(only_prof.shape)[0]
    only_prof= only_prof.sample(n, random_state=42)
    test_only_prof= only_prof.copy()
    only_prof_test= only_prof.tail(1000) # возьмем 1000 профиков для дальнейшего теста
    only_prof=only_prof.head(n-1000)    
    besides_prof=big_data_rfm[big_data_rfm.prof_not==0]
    quantile_10percent_notprof =besides_prof.M.quantile(0.9)
    
    besides_prof =besides_prof[besides_prof['M']<quantile_10percent_notprof]  # убираем 10% самых дорогих по Monetary не профильных карт, это увеличит F1_score примерно на 9%
    besides_prof= besides_prof.sample(n, random_state=0)
    besides_prof_test= besides_prof.tail(1000) # возьмем 1000 не профильных карт для дальнейшего теста
    besides_prof=besides_prof.head(n-1000)
    big_data_rfm= pd.concat([only_prof,besides_prof])   
    big_data_rfm=big_data_rfm.set_index('cart')
    big_data_rfm.index.name = None
    big_data_rfm = big_data_rfm.sample(n=(big_data_rfm.shape)[0], random_state=0) 
    # тесттовая выборка из 2000 карт
    test_2= pd.concat([only_prof_test,besides_prof_test])
    test_2=test_2.set_index('cart')
    test_2.index.name = None
    print('test_2.shape {}'.format(test_2.shape))    
    return big_data_rfm, test_2, test_only_prof

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
    
def get_train_test_sets():
    """Create train and test sets and dump list of cards on wich we learned"""
    big_data_rfm= get_RFM_data()
    big_data_rfm=get_cards_17_digits(big_data_rfm)
    big_data_rfm= get_column_prof_flag(big_data_rfm)
    big_data_rfm, test_2,test_only_prof=get_balance_dataset(big_data_rfm)
    features = ['R','F','M','avg_rec_quant','shop_count','avg_positions_num']
    X = big_data_rfm[features]
    Y = big_data_rfm['prof_not']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, train_size=0.8) 
    #для тестовой выборки 
    X_2 = test_2[features]
    Y_2 = test_2['prof_not']
    X_3 = test_only_prof[features]
    Y_3 = test_only_prof['prof_not']
    scaler = StandardScaler()
    X_2 = scaler.fit_transform(X_2)
    X_3 = scaler.fit_transform(X_3)
    #создаем список карт на которых обучились
    big_data_rfm = big_data_rfm.reset_index(drop=False)
    big_data_rfm= big_data_rfm.rename(columns= {'index':'cart'})
    
    list_of_train = big_data_rfm.cart.unique().tolist() 
    list_of_train= [str(item) for item in list_of_train]
    with open('data/list_of_train.pkl', 'wb') as fid:
        pickle.dump(list_of_train, fid)    
    return X_train, X_test, Y_train, Y_test, X_2, Y_2, X_3, Y_3


def fit_model():
    """ Learn the classifier, prints metrics"""
    X_train, X_test, Y_train, Y_test, X_2, Y_2, X_3, Y_3 = get_train_test_sets()
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
    print("____________________")
    return votingC

def dump_classifier():
    """dump our classifire to pickle"""
    votingC = fit_model()
    with open('data/votingC.pkl', 'wb') as fid:
        pickle.dump(votingC, fid)
    return votingC

# используем обученный классификатор и определяем класс карты

def remove_learned_cart(big_data_rfm):
    """Remove cards wich we learned"""
    
    with open('data/list_of_train.pkl', 'rb') as fid:
        list_of_train = pickle.load(fid)
    big_data_rfm= big_data_rfm[~big_data_rfm['cart'].isin(list_of_train)]
    return big_data_rfm

def get_predict():
    """Defines the class of the customer card"""
    votingC = dump_classifier()
    big_data_rfm= get_RFM_data()
    big_data_rfm = remove_learned_cart(big_data_rfm)
    big_data_rfm=big_data_rfm.set_index('cart')
    big_data_rfm.index.name = None
    features = ['R','F','M','avg_rec_quant','shop_count','avg_positions_num']
    X = big_data_rfm[features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print('start defines the class of the customer card')
    with open('data/votingC.pkl', 'rb') as fid:
        votingC = pickle.load(fid)
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

def save_result_to_csv():
    big_data_rfm = get_predict()
    big_data_rfm = big_data_rfm.reset_index(drop=False)
    big_data_rfm =big_data_rfm.rename(columns= {'index':'cart'})
    big_data_rfm = big_data_rfm[['cart', 'status']]
    big_data_rfm.status = big_data_rfm.status.astype(int)
    print('big_data_rfm:', big_data_rfm.head())
    big_data_rfm.to_csv('data/Classification professional or not.csv', sep=',', encoding='utf-8', index=False)
     
if __name__=='__main__':
    save_result_to_csv()
