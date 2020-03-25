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


def get_RFM_data():
    """ load rfm report"""
    server = 'gp.data.lmru.tech'
    database = 'adb'
    username = '60075437'
    password = 'Passwd321'
    table = 'cards_tic_agg_marts.v_weekly_purchases_by_dept'
    engine = create_engine('postgresql://{}:{}@{}/{}'.format(username, password,  server, database))
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
    print("____________________")
    print("The number of incoming cards in the model {}". format((big_data_rfm.shape)[0]))
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
    only_prof =only_prof[only_prof['M']>quantile_10percent]  # убираем 10% самых низких по Monetary увеличит F1_score примерно на 9%
    n=(only_prof.shape)[0]
    only_prof= only_prof.sample(n, random_state=42)
    test_only_prof= only_prof.copy()
    only_prof_test= only_prof.tail(1000) # возьмем 1000 профиков для дальнейшего теста
    only_prof=only_prof.head(n-1000)    
    besides_prof=big_data_rfm[big_data_rfm.prof_not==0]
    quantile_10percent_notprof =besides_prof.M.quantile(0.9)
    
    besides_prof =besides_prof[besides_prof['M']<quantile_10percent_notprof]  # убираем 10% самых дорогих по Monetary 
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
    return big_data_rfm, test_2, test_only_prof

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

def remove_learned_cart(big_data_rfm):
    """Delete the cards we trained on"""
    
    with open('data/list_of_train.pkl', 'rb') as fid:
        list_of_train = pickle.load(fid)
    big_data_rfm= big_data_rfm[~big_data_rfm['cart'].isin(list_of_train)]
    return big_data_rfm
