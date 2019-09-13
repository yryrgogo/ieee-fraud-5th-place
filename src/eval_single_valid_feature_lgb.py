from glob import glob
import os
from pathlib import Path
import re
import sys
import yaml
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from func.utils import timer, get_categorical_features, read_pkl_gzip, to_pkl_gzip, parallel_load_data, get_filename, logger_func
from ieee_train import eval_train, eval_check_feature
from kaggle_utils import reduce_mem_usage, move_feature

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import shutil

try:
    logger
except NameError:
    logger = logger_func()
    

COLUMN_ID = 'TransactionID'
COLUMN_DT = 'TransactionDT'
COLUMN_TARGET = 'isFraud'
COLUMN_GROUP = 'DT-M'
COLUMNS_IGNORE = [COLUMN_ID, COLUMN_DT, COLUMN_TARGET, COLUMN_GROUP, 'is_train', 'date']

paths_train = glob('../feature/raw_use/*_train.gz')
paths_train += sorted(glob('../feature/org_use/*_train.gz'))

df_train = parallel_load_data(paths_train)

group_kfold_path = '../input/0908_ieee__DT-M_GroupKFold.gz'
group = read_pkl_gzip(group_kfold_path)
df_train[COLUMN_GROUP] = group
df_train = df_train[('2018-1' <= df_train[COLUMN_GROUP]) & (df_train[COLUMN_GROUP] <= '2018-5')]


#========================================================================
# Base Featureに検証用Feature Groupを追加して、スコアの変化を見る.
# Baseより向上したFeature Groupのみ、追加検証を行う
#========================================================================

exp_no = sys.argv[1]
valid_no = sys.argv[2]
if valid_no=='1':
    is_reverse=False
    i_add = 0
elif valid_no=='2':
    is_reverse=True
    i_add = 0
elif valid_no=='3':
    is_reverse=False
    i_add = 20


is_base = [True, False][1]
to_dir = '../feature/check_trush/'
error_dir = '../feature/useless/'
save_file_path = '../output/valid_single_feature.csv'

def get_tree_importance(estimator, use_cols, importance_type="gain"):
    feim = estimator.feature_importance(importance_type=importance_type)
    feim = pd.DataFrame([np.array(use_cols), feim]).T
    feim.columns = ['feature', 'importance']
    feim['importance'] = feim['importance'].astype('float32')
    return feim


valid_paths_train = sorted(glob('../feature/valid/*_train.gz'), reverse=is_reverse)
for i in range(10):

    #  valid_path = valid_paths_train[(i+i_add):(i+i_add)+1]
    valid_path = valid_paths_train[(i+i_add)*10:(i+i_add+1)*10]
    try:
        df_feat_train = parallel_load_data(valid_path)
        cols_feat = df_feat_train.columns
        cols_feat = ['z__' + col for col in cols_feat]
        df_feat_train.column = cols_feat
    except EOFError:
        for path in valid_path:
            try:
                shutil.move(path, error_dir)
            except FileNotFoundError:
                print(path)
        continue


    if is_base:
        tmp_train = df_train.copy()
        feature_name = 'base'
    else:
        tmp_train = df_train.join(df_feat_train)
        feature_name = get_filename(valid_path[0])
    
    use_cols = [col for col in tmp_train.columns if col not in COLUMNS_IGNORE]
        
    for fold in range(2):

        with timer('  * Make Dataset'):
            if fold==0:
                dataset = tmp_train[
                    (tmp_train[COLUMN_GROUP] == '2018-2') | 
                    (tmp_train[COLUMN_GROUP] == '2018-3') | 
                    (tmp_train[COLUMN_GROUP] == '2018-5')]
                train = dataset[('2018-2' <= dataset[COLUMN_GROUP]) & (dataset[COLUMN_GROUP] <= '2018-3')]
                test  = dataset[dataset[COLUMN_GROUP] == '2018-5']
            elif fold==1:
                dataset = tmp_train[
                    (df_train[COLUMN_GROUP] == '2018-1') | 
                    (df_train[COLUMN_GROUP] == '2018-2') | 
                    (df_train[COLUMN_GROUP] == '2018-4')]
                train = dataset[('2018-1' <= dataset[COLUMN_GROUP]) & (dataset[COLUMN_GROUP] <= '2018-2')]
                test  = dataset[dataset[COLUMN_GROUP] == '2018-4']
        
            Y_TRAIN = train[COLUMN_TARGET]
            train.drop(COLUMN_TARGET, axis=1, inplace=True)
        
            Y_TEST = test[COLUMN_TARGET]
            test.drop(COLUMN_TARGET, axis=1, inplace=True)
        
        start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())[:13]
        params = {
            'n_jobs': 20,
            'seed': 1208,
            'metric': 'auc',
            'objective': 'binary',
            'num_leaves': 2**7-1,
            'max_depth': -1,
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree' : 1.0,
            'lambda_l1' : 0.1,
            'lambda_l2' : 1.0,
            'learning_rate' : 0.1,
        }
        
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
            logger.info(f"  * {feature_name} Fold{fold}:{score}")
            for path in valid_path:
                feature_name = get_filename(path)
                with open(save_file_path, 'a') as f:
                    line = f'{exp_no},{fold},{feature_name},{score}\n'
                    f.write(line)

#             feim = get_tree_importance(estimator=estimator, use_cols=x_train.columns)
#             feim.sort_values(by='importance', ascending=False, inplace=True)
#             feim['is_valid'] = feim['feature'].map(valid_map)

    if is_base:
        sys.exit()
        
    #========================================================================
    # PostProcess
    #========================================================================
    with timer("  * PostProcess"):
        for path in valid_path:
            try:
                shutil.move(path, to_dir)
            except FileNotFoundError:
                print(feature_name)
