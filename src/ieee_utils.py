import os
import sys
import yaml
import datetime
import numpy as np
import pandas as pd
from func.utils import get_categorical_features
from func.ml_utils import Classifier
from func.BigQuery import BigQuery
from sklearn.model_selection import StratifiedKFold
from bq_log import create_train_log_table, save_train_log


COLUMN_ID = 'TransactionID'
COLUMN_TARGET = 'isFraud'
COLUMNS_IGNORE = [COLUMN_ID, COLUMN_TARGET]

train_df = pd.read_csv('../input/train_transaction.csv')
test_df  = pd.read_csv('../input/test_transaction.csv')
Y = train_df[COLUMN_TARGET]


def first_exec():
    create_train_log_table()


def ieee_train(train_df, test_df, params={}):
    start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())[:13]
    seed       = params['seed']
    model_type = params['model_type']
    del params['model_type']
    
    columns_category = get_categorical_features(train_df, COLUMNS_IGNORE)
    
    kfold = StratifiedKFold(n_splits=n_splits, random_state=seed)
    kfold = list(kfold.split(train_df, Y))
    score_list = []
    feim_list = []
    y_pred = np.zeros(len(train_df))
    test_preds = []
    use_cols = [col for col in train_df.columns if col not in COLUMNS_IGNORE + columns_category]
    
    for n_fold, (trn_idx, val_idx) in enumerate(kfold):
        x_train = train_df.iloc[trn_idx][use_cols]
        y_train = Y.iloc[trn_idx]
        x_valid = train_df.iloc[val_idx][use_cols]
        y_valid = Y.iloc[val_idx]
    
        score, oof_pred, test_pred, feim, _ = Classifier(
            model_type=model_type,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            x_test=test_df[use_cols],
            params=params,
        )

        score_list.append(score)
        y_pred[val_idx] = oof_pred
        test_preds.append(test_pred)
        
        feim.rename(columns={'importance': f'importance_fold{n_fold+1}'}, inplace=True)
        feim.set_index('feature', inplace=True)
        feim_list.append(feim)
        
    cv_score = np.mean(score_list)
    feim_df = pd.concat(feim_list, axis=1)
    feim_df['importance_avg'] = feim_df.mean(axis=1)
    feim_df.sort_values(by='importance_avg', ascending=False, inplace=True)
    
    test_pred_avg = np.mean(test_preds, axis=1)
    all_pred = np.append(y_pred, test_pred_avg)
    all_ids = np.append(train_df[COLUMN_ID].values, test_df[COLUMN_ID].values)
    pred_result = pd.Series(all_pred, index=all_ids, name='pred_' + start_time)

    #========================================================================
    # Save
    #========================================================================
    # Feature Importance
    to_pkl_gzip(obj=feim_df, path=f"../output/feature_importances/{start_time}__CV{str(cv_score).replace('.', '-')}__feature{len(use_cols)}")
    # Prediction
    to_pkl_gzip(obj=pred_result, path=f"../output/pred_result/{start_time}__CV{str(cv_score).replace('.', '-')}__all_preds")
    
    #========================================================================
    # Log
    #========================================================================
    log_map = {}
    log_map['exp_date']    = start_time
    log_map['n_features']  = train_df.shape[1]
    log_map['n_rows']      = train_df.shape[0]
    log_map['cv_score']    = cv_score
    log_map['fold1_score'] = score_list[0]
    log_map['fold2_score'] = score_list[1]
    log_map['fold3_score'] = score_list[2]
    log_map['fold4_score'] = score_list[3]
    log_map['fold5_score'] = score_list[4]
    log_map['seed']        = seed
    log_map['metric']      = params['metric']
    log_map['model_type']  = model_type
    save_train_log(log_map, params)
    
    
def get_params():
    metric     = 'auc'
    model_type = 'lgb'
    n_splits   = 5