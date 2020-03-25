import pandas as pd
import preprocessing as preprocessing
import train as train
import class_predict as clp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('always')

def save_result_to_csv():
    
    X_train, X_test, Y_train, Y_test, X_2, Y_2, X_3, Y_3=  preprocessing.get_train_test_sets()
    votingC =train.dump_classifier(X_train, X_test, Y_train, Y_test, X_2, Y_2, X_3, Y_3)
    big_data_rfm= clp.get_predict(votingC)
    big_data_rfm = big_data_rfm.reset_index(drop=False)
    big_data_rfm =big_data_rfm.rename(columns= {'index':'cart'})
    big_data_rfm = big_data_rfm[['cart', 'status']]
    big_data_rfm.loc['status'] = big_data_rfm['status'].astype(int)
    big_data_rfm.loc[:,'cart'] =  big_data_rfm['cart'].astype(str)
    print( big_data_rfm.head())
    big_data_rfm.to_csv('predictions/Classification professional or not.csv', sep=',', encoding='utf-8', index=False)
     
if __name__=='__main__':
    save_result_to_csv()
