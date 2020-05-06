import pickle
import preprocessing as pr
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('always')


def get_predict():
    """Defines the class of the customer card"""

    big_data_rfm = pr.get_RFM_data()
    big_data_rfm = pr.remove_learned_cart(big_data_rfm)
    big_data_rfm = big_data_rfm.set_index('cart')
    big_data_rfm.index.name = None
    features = [
        'R', 'F', 'shop_count', 'M', 'avg_rec_quant', 'avg_positions_num',
        'avg_week', 'avg_count_week', 'avg_count_dept_year',
        'avg_sum_revenue_year', 'coef_quant', 'coef_M'
    ]
    X = big_data_rfm[features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print('start defines the class of the customer card')
    with open('data/votingC.pkl', 'rb') as fid:
        votingC = pickle.load(fid)
    predictions = (votingC.predict_proba(X)[:, 1] >= 0.8).astype(
        bool)  # задается порог определения класса
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
    big_data_rfm = big_data_rfm.rename(columns={'index': 'cart'})
    big_data_rfm = big_data_rfm[['cart', 'status']]
    big_data_rfm.loc['status'] = big_data_rfm['status'].astype(int)
    big_data_rfm.loc[:, 'cart'] = big_data_rfm['cart'].astype(str)
    print(big_data_rfm.head())
    big_data_rfm.to_csv('data/Classification professional or not.csv',
                        sep=',',
                        encoding='utf-8',
                        index=False)


if __name__ == '__main__':
    save_result_to_csv()
