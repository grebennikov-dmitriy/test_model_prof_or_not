import pickle
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
import loader as ld
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('always')


def get_RFM_data():
    """ load rfm report"""
    big_data_rfm = ld.merge_rfm_and_other_features()
    big_data_rfm = big_data_rfm.rename(columns={
        'cod_cartfid': 'cart',
        'r': 'R',
        'm': 'M',
        'f': 'F'
    })

    # Загрузим данные по чекам
    with open('data/avg_rec_per_user_part1.pkl', 'rb') as fid:
        avg_rec_per_user_part1 = pickle.load(fid)
    with open('data/avg_rec_per_user_part2.pkl', 'rb') as fid:
        avg_rec_per_user_part2 = pickle.load(fid)
    avg_rec_per_user = pd.concat(
        [avg_rec_per_user_part1, avg_rec_per_user_part2])
    big_data_rfm = big_data_rfm.merge(avg_rec_per_user, on='cart')
    # отношение средних за неделю к средним за год
    big_data_rfm['coef_quant'] = big_data_rfm['avg_count_week'] / big_data_rfm[
        'avg_count_dept_year']
    big_data_rfm['coef_M'] = big_data_rfm['avg_week'] / big_data_rfm[
        'avg_sum_revenue_year']
    big_data_rfm = big_data_rfm.fillna(0)
    print("____________________")
    print("The number of incoming cards in the model {}".format(
        (big_data_rfm.shape)[0]))
    return big_data_rfm


# Оставим карты с 17-ю цифрами
def get_cards_17_digits(big_data_rfm):
    """ get cards with 17 digits"""
    big_data_rfm.loc[:, 'len_cart'] = big_data_rfm.cart.apply(lambda x: len(x))
    big_data_rfm = big_data_rfm[big_data_rfm.len_cart == 17]
    return big_data_rfm


def assign_prof_cards(cart):
    """True if the 8th digit in the card is 3, otherwise False """
    list1 = [cart]
    result = [j for i in list1 for j in i]
    key_number = result[7]
    if key_number == '3':
        x = True
    else:
        x = False
    return x


def get_column_prof_flag(big_data_rfm):
    big_data_rfm.loc[:, 'prof_not'] = big_data_rfm.cart.apply(assign_prof_cards)
    del big_data_rfm['len_cart']
    return big_data_rfm


def get_balance_dataset(big_data_rfm):
    """Сreates a class-balanced dataset"""
    only_prof = big_data_rfm[big_data_rfm.prof_not == 1]
    quantile_10percent = only_prof.M.quantile(0.1)
    only_prof = only_prof[
        only_prof['M'] >
        quantile_10percent]  # убираем 10% самых низких по Monetary увеличит F1_score примерно на 9%
    n = (only_prof.shape)[0]
    only_prof = only_prof.sample(n, random_state=42)
    test_only_prof = only_prof.copy()
    only_prof_test = only_prof.tail(
        1000)  # возьмем 1000 профиков для дальнейшего теста
    only_prof = only_prof.head(n - 1000)
    besides_prof = big_data_rfm[big_data_rfm.prof_not == 0]
    quantile_10percent_notprof = besides_prof.M.quantile(0.9)

    besides_prof = besides_prof[
        besides_prof['M'] <
        quantile_10percent_notprof]  # убираем 10% самых дорогих по Monetary
    besides_prof = besides_prof.sample(n, random_state=0)
    besides_prof_test = besides_prof.tail(
        1000)  # возьмем 1000 не профильных карт для дальнейшего теста
    besides_prof = besides_prof.head(n - 1000)
    big_data_rfm = pd.concat([only_prof, besides_prof])
    big_data_rfm = big_data_rfm.set_index('cart')
    big_data_rfm.index.name = None
    big_data_rfm = big_data_rfm.sample(n=(big_data_rfm.shape)[0],
                                       random_state=0)
    # тесттовая выборка из 2000 карт
    test_2 = pd.concat([only_prof_test, besides_prof_test])
    test_2 = test_2.set_index('cart')
    test_2.index.name = None
    return big_data_rfm, test_2, test_only_prof


def get_train_test_sets():
    """Create train and test sets and dump list of cards on wich we learned"""
    big_data_rfm = get_RFM_data()
    big_data_rfm = get_cards_17_digits(big_data_rfm)
    big_data_rfm = get_column_prof_flag(big_data_rfm)
    big_data_rfm, test_2, test_only_prof = get_balance_dataset(big_data_rfm)
    features = [
        'R', 'F', 'shop_count', 'M', 'avg_rec_quant', 'avg_positions_num',
        'avg_week', 'avg_count_week', 'avg_count_dept_year',
        'avg_sum_revenue_year', 'coef_quant', 'coef_M'
    ]

    X_train = big_data_rfm[features]
    Y_train = big_data_rfm['prof_not']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # для тестовой выборки
    X_2 = test_2[features]
    Y_2 = test_2['prof_not']
    X_3 = test_only_prof[features]
    Y_3 = test_only_prof['prof_not']
    X_2 = scaler.fit_transform(X_2)
    X_3 = scaler.fit_transform(X_3)
    # создаем список карт на которых обучились
    big_data_rfm = big_data_rfm.reset_index(drop=False)
    big_data_rfm = big_data_rfm.rename(columns={'index': 'cart'})
    list_of_train = big_data_rfm.cart.unique().tolist()
    list_of_train = [str(item) for item in list_of_train]
    with open('data/list_of_train.pkl', 'wb') as fid:
        pickle.dump(list_of_train, fid)
    return X_train, Y_train, X_2, Y_2, X_3, Y_3


def remove_learned_cart(big_data_rfm):
    """Delete the cards we trained on"""

    with open('data/list_of_train.pkl', 'rb') as fid:
        list_of_train = pickle.load(fid)
    big_data_rfm = big_data_rfm[~big_data_rfm['cart'].isin(list_of_train)]
    return big_data_rfm
