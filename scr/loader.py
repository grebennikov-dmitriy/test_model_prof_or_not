import pandas as pd
from sqlalchemy import create_engine

sql_query_avg = """ select distinct cod_cartfid , 
        avg(sum(revenue)) over (partition by cod_cartfid ) as avg_week,
        avg(count(revenue)) over (partition by cod_cartfid ) as avg_count_week
FROM cards_tic_agg_marts.v_weekly_purchases_by_dept
group by   cod_cartfid, date_trunc('week', monday_of_week), dept  """

sql_query_quarter = """select distinct cod_cartfid,
        avg(sum(revenue)) over (partition by cod_cartfid ) as avg_sum_year,
        avg(count(dept)) over (partition by cod_cartfid ) as avg_count_year,
        extract(quarter from monday_of_week) as quarter
FROM cards_tic_agg_marts.v_weekly_purchases_by_dept
group by   cod_cartfid, date_trunc('year', monday_of_week), dept, quarter   """

sql_query = """ SELECT cod_cartfid ,
       current_date -max(monday_of_week) as R,
       count(DISTINCT monday_of_week) as F,
       SUM(revenue) AS M,
       COUNT(DISTINCT num_ett) AS shop_count,
       avg(positions_num) as avg_positions_num
FROM cards_tic_agg_marts.v_weekly_purchases_by_dept
group by cod_cartfid  """


def engine_for_greenplum(host='gp.data.lmru.tech',
                         dbname='adb',
                         user='60075437',
                         password=None):
    password = password or open('untitled.txt', 'r').read()
    engine = create_engine(f'postgres://{user}:{password}@{host}/{dbname}')
    return engine


def load_data_from_db():
    """ load rfm report"""
    table = 'cards_tic_agg_marts.v_weekly_purchases_by_dept'
    engine = engine_for_greenplum()
    sql = sql_query.format(table)
    data = pd.read_sql(sql, engine)
    return data


def load_data_quarter():
    """ Load average values         for a year from quarterly averages"""
    table = 'cards_tic_agg_marts.v_weekly_purchases_by_dept'
    engine = engine_for_greenplum()
    sql = sql_query_quarter.format(table)
    data_quarter = pd.read_sql(sql, engine)
    data_quarter.set_index(['cod_cartfid', 'quarter'], inplace=True)
    data_quarter = data_quarter.unstack()
    data_quarter = data_quarter.reset_index()
    data_quarter['avg_count_dept_year'] = data_quarter['avg_count_year'].mean(
        axis=1)
    data_quarter['avg_sum_revenue_year'] = data_quarter['avg_sum_year'].mean(
        axis=1)
    # убираем multiIndex
    data_quarter = data_quarter.drop(['avg_count_year', 'avg_sum_year'],
                                     axis=1,
                                     level=0)
    data_quarter.columns.name = None
    data_quarter.reset_index()
    data_quarter = data_quarter.set_index(
        ['cod_cartfid', 'avg_count_dept_year', 'avg_sum_revenue_year'])
    data_quarter.columns = data_quarter.columns.droplevel(0)
    data_quarter = data_quarter.reset_index()
    data_quarter = data_quarter.rename_axis(index=None, columns=None)
    return data_quarter


def load_avg_week():
    """ loads weekly average """
    table = 'cards_tic_agg_marts.v_weekly_purchases_by_dept'
    engine = engine_for_greenplum()
    sql = sql_query_avg.format(table)
    aver_by_dept = pd.read_sql(sql, engine)
    return aver_by_dept


def merge_rfm_and_other_features():
    data = load_data_from_db()
    aver_by_dept = load_avg_week()
    data_quarter = load_data_quarter()
    big_data_rfm = pd.merge(data, aver_by_dept, on='cod_cartfid')
    big_data_rfm = pd.merge(big_data_rfm, data_quarter, on='cod_cartfid')

    return big_data_rfm


if __name__ == '__main__':
    merge_rfm_and_other_features()
