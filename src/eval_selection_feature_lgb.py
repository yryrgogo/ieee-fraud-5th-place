from glob import glob
from tqdm import tqdm
import datetime
import numpy as np
import os
import pandas as pd
import re
import sys

from func.utils import timer, read_pkl_gzip, to_pkl_gzip, parallel_load_data, get_filename
from kaggle_utils import move_feature

if sys.argv[1]=='1':
    is_reverse = False
    i_add = 0
elif sys.argv[1]=='2':
    is_reverse = True
    i_add = 0
elif sys.argv[1]=='3':
    is_reverse = True
    i_add = 20

error_dir = '../feature/check_trush/'
to_dir = '../feature/valid_trush/'
COLUMN_ID = 'TransactionID'
COLUMN_DT = 'TransactionDT'
COLUMN_TARGET = 'isFraud'
COLUMN_GROUP = 'DT-M'
COLUMNS_IGNORE = [COLUMN_ID, COLUMN_DT, COLUMN_TARGET, COLUMN_GROUP, 'is_train', 'date']

paths_train = glob('../feature/raw_use/*_train.gz')
paths_train += sorted(glob('../feature/org_use/*_train.gz'))
paths_train += sorted(glob('../feature/kernel/*_train.gz'))

df_train = parallel_load_data(paths_train)

group_kfold_path = '../input/0908_ieee__DT-M_GroupKFold.gz'
group = read_pkl_gzip(group_kfold_path)
df_train[COLUMN_GROUP] = group

df_train = df_train[('2018-2' == df_train[COLUMN_GROUP]) | (df_train[COLUMN_GROUP] == '2018-3') | (df_train[COLUMN_GROUP] == '2018-5')]


#========================================================================
# Base Featureに検証用Feature Groupを追加して、スコアの変化を見る.
# Baseより向上したFeature Groupのみ、追加検証を行う
#========================================================================

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import shutil

def get_tree_importance(estimator, use_cols, importance_type="gain"):
    feim = estimator.feature_importance(importance_type=importance_type)
    feim = pd.DataFrame([np.array(use_cols), feim]).T
    feim.columns = ['feature', 'importance']
    feim['importance'] = feim['importance'].astype('float32')
    return feim


#========================================================================
# 10loop以上するとtoo many open fileでエラーになる
#========================================================================
for i in range(10):
    valid_paths_train = sorted(glob('../feature/valid/*_train.gz'), reverse=is_reverse)[int(i+i_add)*10:int(i+i_add+1)*10]
#     valid_paths_train = sorted(glob('../feature/valid/*_train.gz'), reverse=is_reverse)
    
    valid_map = {}
    for path in valid_paths_train:
        filename = re.search(r'/([^/.]*).gz', path).group(1)
        valid_map[filename.replace('_train', '')] = 1
    
    with timer('  * Make Dataset'):

        try:
            df_feat_train = parallel_load_data(valid_paths_train)
        except EOFError:
            for path in valid_paths_train:
                try:
                    shutil.move(path, error_dir)
                except FileNotFoundError:
                    print(path)
            continue
        
        tmp_train = df_train.join(df_feat_train)
        train = tmp_train[('2018-2' <= tmp_train[COLUMN_GROUP]) & (tmp_train[COLUMN_GROUP] <= '2018-3')]
        Y_TRAIN = train[COLUMN_TARGET]
        train.drop(COLUMN_TARGET, axis=1, inplace=True)
    
        test  = tmp_train[tmp_train[COLUMN_GROUP] == '2018-5']
        Y_TEST = test[COLUMN_TARGET]
        test.drop(COLUMN_TARGET, axis=1, inplace=True)
    
    start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())[:13]
    params = {
        'n_jobs': 31,
        'seed': 1208,
        'metric': 'auc',
        'objective': 'binary',
        'num_leaves': 2**8-1,
        'max_depth': -1,
        'subsample': 0.9,
        'subsample_freq': 1,
        'colsample_bytree' : 0.20,
        'lambda_l1' : 0.1,
        'lambda_l2' : 1.0,
        'learning_rate' : 0.1,
    }
    
    use_cols = [col for col in tmp_train.columns if col not in COLUMNS_IGNORE]
    x_train = train[use_cols]
    y_train = Y_TRAIN
    x_valid = test[use_cols]
    y_valid = Y_TEST
    early_stopping_rounds=20
    num_boost_round=3500
    metric = 'auc'
    params['metric'] = metric
    
    #========================================================================
    # Fitting
    #========================================================================
    lgb_train = lgb.Dataset(data=x_train, label=y_train)
    lgb_valid = lgb.Dataset(data=x_valid, label=y_valid)
    
    with timer("  * Train & Validation"):
        estimator = lgb.train(
            params = params,
            train_set = lgb_train,
            valid_sets = lgb_valid,
            early_stopping_rounds = early_stopping_rounds,
            num_boost_round = num_boost_round,
            verbose_eval = 200
        )
        best_iter = estimator.best_iteration
    
        oof_pred = estimator.predict(x_valid)
        score = roc_auc_score(y_valid, oof_pred)
        cvs = str(score).replace('.', '-')
        feim = get_tree_importance(estimator=estimator, use_cols=x_train.columns)
        feim.sort_values(by='importance', ascending=False, inplace=True)
        feim['is_valid'] = feim['feature'].map(valid_map)
    
    #========================================================================
    # PostProcess
    #========================================================================
    
    with timer("  * PostProcess"):
        to_pkl_gzip(obj=feim, path=f"../output/selection_feature/{start_time}__CV{cvs}__feature{len(use_cols)}")
        for path in valid_paths_train:
            try:
                shutil.move(path, to_dir)
                shutil.move(path.replace('train', 'test'), to_dir)
            except FileNotFoundError:
                print(feature_name)
