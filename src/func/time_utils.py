import numpy as np
import pandas as pd
import sys
import re
import gc
import glob
import datetime
from dateutil.parser import parse
from datetime import date, timedelta


def diff_of_days(day1, day2):
    try:
        days = (parse(day1) - parse(day2)).days
    except TypeError:
        days = (day1 - day2).days
    return days

def diff_of_times(day1, day2):
    try:
        days = (parse(day1) - parse(day2))
    except TypeError:
        days = (day1 - day2)
    return days

def date_add_days(start_date, add_days):
    if str(type(start_date)).count('time'):
        end_date = start_date + timedelta(days=add_days)
    elif str(type(start_date)).count('str'):
        end_date = parse(start_date) + timedelta(days=add_days)
        end_date = end_date.strftime('%Y%m%d')
    return end_date

def date_add_times(start_date, add_times):
    if str(type(start_date)).count('time'):
        end_date = start_date + timedelta(hours=add_times)
    elif str(type(start_date)).count('str'):
        end_date = parse(start_date) + timedelta(hours=add_times)
        end_date = end_date.strftime('%Y-%m-%d-%H')
    return end_date

def get_label(df, end_date, n_day):
    start_date = date_add_days(end_date, -1*n_day)
    label = df[(df['date'] < end_date) & (
        df['date'] >= start_date)]['date'].values
    #  label['start_date'] = start_date
    #  label['diff_of_day'] = label['date'].apply(lambda x: diff_of_days(x, start_date))
    #  label['month'] = label['date'].str[4:6].astype(int)
    #  label['year'] = label['date'].str[:4].astype(int)
    #  label = label.reset_index(drop=True)
    return label

def get_select_term_feat(df, base_key, base_date, end_date, n_day, value_name):
    """Summary line.
    end_date = base_dateとして使うのがメインだが、時々ずらしたいことがあるので
    base_dateを別途用意している

    Args:

    Returns:
    """
    start_date = date_add_days(end_date, -n_day)
    df_tmp = df[ (start_date <= df.date) & (df.date <= end_date) ].copy()
    logger.info(f'''
#========================================================================
# base_key   : {base_key}
# base_date  : {base_date}
# start_date : {start_date}
# end_date   : {end_date}
# n_day      : {n_day}
# value_name : {value_name}
#========================================================================''')

    result = df_tmp.groupby([base_key], as_index=False)[value_name].agg({f'{base_key}_min_day{n_day}': 'min',
                                                                           f'{base_key}_mean_day{n_day}': 'mean',
                                                                           f'{base_key}_median_day{n_day}': 'median',
                                                                           f'{base_key}_max_day{n_day}': 'max',
                                                                           f'{base_key}_count_day{n_day}': 'count',
                                                                           f'{base_key}_std_day{n_day}': 'std',
                                                                           f'{base_key}_skew_day{n_day}': 'skew'})
    result.reset_index(inplace=True)
    result['date'] = base_date
    return result

def get_country_exp_visitor_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    df_temp = df[(df.visit_date < key[0]) & (
        df.visit_date > start_date)].copy()
    df_temp['visit_date'] = df_temp['visit_date'].apply(
        lambda x: diff_of_days(key[0], x))
    df_temp['weight'] = df_temp['visit_date'].apply(lambda x: 0.985**x)
    df_temp['visitors'] = df_temp['visitors'] * df_temp['weight']
    result1 = df_temp.groupby(['country_id'], as_index=False)['visitors'].agg({
        'country_exp_mean{}'.format(n_day): 'sum'})
    result2 = df_temp.groupby(['country_id'], as_index=False)['weight'].agg(
        {'country_exp_weight_sum{}'.format(n_day): 'sum'})
    result = result1.merge(result2, on=['country_id'], how='left')
    result['country_exp_mean{}'.format(n_day)] = result['country_exp_mean{}'.format(
        n_day)]/result['country_exp_weight_sum{}'.format(n_day)]
    result = left_merge(label, result, on=['country_id']).fillna(0)
    return result


def get_country_week_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    df_temp = df[(df.visit_date < key[0]) & (
        df.visit_date > start_date)].copy()
    result = df_temp.groupby(['country_id', 'dow'], as_index=False)['visitors'].agg({'country_dow_min{}'.format(n_day): 'min',
                                                                                       'country_dow_mean{}'.format(n_day): 'mean',
                                                                                       'country_dow_median{}'.format(n_day): 'median',
                                                                                       'country_dow_max{}'.format(n_day): 'max',
                                                                                       'country_dow_count{}'.format(n_day): 'count',
                                                                                       'country_dow_std{}'.format(n_day): 'std',
                                                                                       'country_dow_skew{}'.format(n_day): 'skew'})
    result = left_merge(label, result, on=['country_id', 'dow']).fillna(0)
    return result


def get_country_week_diff_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    df_temp = df[(df.visit_date < key[0]) & (
        df.visit_date > start_date)].copy()
    result = df_temp.set_index(['country_id', 'visit_date'])[
        'visitors'].unstack()
    result = result.diff(axis=1).iloc[:, 1:]
    c = result.columns
    result['country_diff_mean'] = np.abs(result[c]).mean(axis=1)
    result['country_diff_std'] = result[c].std(axis=1)
    result['country_diff_max'] = result[c].max(axis=1)
    result['country_diff_min'] = result[c].min(axis=1)
    result = left_merge(label, result[['country_diff_mean', 'country_diff_std',
                                       'country_diff_max', 'country_diff_min']], on=['country_id']).fillna(0)
    return result


def get_country_all_week_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    df_temp = df[(df.visit_date < key[0]) & (
        df.visit_date > start_date)].copy()
    result_temp = df_temp.groupby(['country_id', 'dow'], as_index=False)['visitors'].agg({'country_dow_mean{}'.format(n_day): 'mean',
                                                                                            'country_dow_median{}'.format(n_day): 'median',
                                                                                            'country_dow_sum{}'.format(n_day): 'max',
                                                                                            'country_dow_count{}'.format(n_day): 'count'})
    result = pd.DataFrame()
    for i in range(7):
        result_sub = result_temp[result_temp['dow'] == i].copy()
        result_sub = result_sub.set_index('country_id')
        result_sub = result_sub.add_prefix(str(i))
        result_sub = left_merge(label, result_sub, on=['country_id']).fillna(0)
        result = pd.concat([result, result_sub], axis=1)
    return result


def get_country_week_exp_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    df_temp = df[(df.visit_date < key[0]) & (
        df.visit_date > start_date)].copy()
    df_temp['visit_date'] = df_temp['visit_date'].apply(
        lambda x: diff_of_days(key[0], x))
    df_temp['visitors2'] = df_temp['visitors']
    result = None
    for i in [0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.999, 0.9999]:
        df_temp['weight'] = df_temp['visit_date'].apply(lambda x: i**x)
        df_temp['visitors1'] = df_temp['visitors'] * df_temp['weight']
        df_temp['visitors2'] = df_temp['visitors2'] * df_temp['weight']
        result1 = df_temp.groupby(['country_id', 'dow'], as_index=False)[
            'visitors1'].agg({'country_dow_exp_mean{}_{}'.format(n_day, i): 'sum'})
        result3 = df_temp.groupby(['country_id', 'dow'], as_index=False)[
            'visitors2'].agg({'country_dow_exp_mean2{}_{}'.format(n_day, i): 'sum'})
        result2 = df_temp.groupby(['country_id', 'dow'], as_index=False)['weight'].agg(
            {'country_dow_exp_weight_sum{}_{}'.format(n_day, i): 'sum'})
        result_temp = result1.merge(
            result2, on=['country_id', 'dow'], how='left')
        result_temp = result_temp.merge(
            result3, on=['country_id', 'dow'], how='left')
        result_temp['country_dow_exp_mean{}_{}'.format(n_day, i)] = result_temp['country_dow_exp_mean{}_{}'.format(
            n_day, i)]/result_temp['country_dow_exp_weight_sum{}_{}'.format(n_day, i)]
        result_temp['country_dow_exp_mean2{}_{}'.format(n_day, i)] = result_temp['country_dow_exp_mean2{}_{}'.format(
            n_day, i)]/result_temp['country_dow_exp_weight_sum{}_{}'.format(n_day, i)]
        if result is None:
            result = result_temp
        else:
            result = result.merge(
                result_temp, on=['country_id', 'dow'], how='left')
    result = left_merge(label, result, on=['country_id', 'dow']).fillna(0)
    return result
