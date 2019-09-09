from bq_log import create_train_log_table, save_train_log
from func.BigQuery import BigQuery
from func.ml_utils import Classifier, display_importance, drop_unique_feature, drop_high_corr_feature
from func.utils import get_categorical_features, to_pkl_gzip, timer
from kaggle_utils import move_feature
from glob import glob
from sklearn.model_selection import StratifiedKFold, GroupKFold
import datetime
import numpy as np
import os
import pandas as pd
import re
import sys
import warnings
import yaml
warnings.simplefilter('ignore')

#========================================================================
# Config
#========================================================================
COLUMN_ID = 'TransactionID'
COLUMN_DT = 'TransactionID'
COLUMN_TARGET = 'isFraud'
COLUMNS_IGNORE = [COLUMN_ID, COLUMN_DT, COLUMN_TARGET, 'is_train', 'pred_user', 'DT-M']
early_stopping_rounds = 50


def join_same_user(df, pred_user_path):
    pred_user = pd.read_csv(pred_user_path)
    pred_user['same_user_id'] = pred_user['predicted_user_id']
    if pred_user['predicted_user_id'].isnull().sum():
        pred_user.loc[pred_user[pred_user['predicted_user_id'].isnull()].index, 'same_user_id'] = \
        pred_user.loc[pred_user[pred_user['predicted_user_id'].isnull()].index, COLUMN_ID]
    
    pred_user['same_user_id'] = pred_user['same_user_id'].astype('int')
    pred_user.set_index(COLUMN_ID, inplace=True)
    
    df.set_index(COLUMN_ID, inplace=True)
    df['pred_user'] = pred_user['same_user_id']
    df.reset_index(inplace=True)
    
    return df

def get_params(model_type):
    params = {
        'n_jobs': 40,
        'seed': 1208,
        'n_splits': 5,
        'metric': 'auc',
        'model_type': model_type,
        'objective': 'binary',
        'fold': ['stratified', 'group'][1],

        'num_leaves': 2**6-1,
        'max_depth': -1,
        'subsample': 1.0,
        'subsample_freq': 1,
        'colsample_bytree' : 0.25,
        'lambda_l1' : 0.1,
        'lambda_l2' : 1.0,
        'learning_rate' : 0.1,
    }
    return params


def ieee_cv(df_train, Y, df_test, COLUMN_GROUP, use_cols, params={}, is_adv=False, is_valid=False):
    start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())[:13]
    seed       = params['seed']
    model_type = params['model_type']
    n_splits = params['n_splits']
    validation = params['fold']
    early_stopping_rounds = params['early_stopping_rounds']
#     del params['model_type'], params['n_splits'], params['fold']
    
    if validation=="stratified":
        kfold = list(StratifiedKFold(n_splits=n_splits, random_state=seed).split(df_train, Y))
    elif validation=='group':
        kfold = list(GroupKFold(n_splits=n_splits).split(df_train, Y, df_train[COLUMN_GROUP]))
        
    score_list = []
    feim_list = []
    best_iteration = 0
    y_pred = np.zeros(len(df_train))
    test_preds = []
    
    if len(df_test):
        x_test = df_test[use_cols]
    else:
        x_test = []
    
    for n_fold, (trn_idx, val_idx) in enumerate(kfold):
        x_train = df_train.iloc[trn_idx][use_cols]
        y_train = Y.iloc[trn_idx]
        x_valid = df_train.iloc[val_idx][use_cols]
        y_valid = Y.iloc[val_idx]
        
        val_gr = df_train.iloc[val_idx][COLUMN_GROUP].value_counts()
        print("="*20)
        with timer(f"  * Fold{n_fold} Validation-{COLUMN_GROUP} {val_gr.index[0]}: {val_gr.values[0]}"):
            score, oof_pred, test_pred, feim, best_iter, _ = Classifier(
                model_type=model_type,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                x_test=x_test,
                params=params,
                early_stopping_rounds = early_stopping_rounds,
            )
        print("="*20)

        score_list.append(score)
        best_iteration += best_iter/n_splits
        y_pred[val_idx] = oof_pred
        test_preds.append(test_pred)
        
        feim.rename(columns={'importance': f'imp_fold{n_fold+1}'}, inplace=True)
        feim.set_index('feature', inplace=True)
        feim_list.append(feim)
        
    cv_score = np.mean(score_list)
    df_feim = pd.concat(feim_list, axis=1)
    df_feim['imp_avg'] = df_feim.mean(axis=1)
    df_feim.sort_values(by='imp_avg', ascending=False, inplace=True)
    
    
    #========================================================================
    # Adversarial Validationもこちらの関数を使うので、Adversarialの場合はここで終わり
    #========================================================================
    if is_adv:
        pred_result = pd.Series(y_pred, index=df_train[COLUMN_ID].values, name='adv_pred_' + start_time)
        return cv_score, df_feim, pred_result
    
    # Feature Importance
    cvs = str(cv_score).replace('.', '-')
    to_pkl_gzip(obj=df_feim, path=f"../output/feature_importances/{start_time}__CV{cvs}__feature{n_features}")
    
    with timer("  * Make Prediction Result File."):
        if is_valid:
            pred_result = []
        else:
            test_pred_avg = np.mean(test_preds, axis=0)
            all_pred = np.append(y_pred, test_pred_avg)
            all_ids = np.append(df_train[COLUMN_ID].values, df_test[COLUMN_ID].values)
            pred_result = pd.DataFrame([all_ids, all_pred], index=[COLUMN_ID, 'pred_' + start_time]).T
            pred_result[COLUMN_ID] = pred_result[COLUMN_ID].astype('int')
            
            #========================================================================
            # Save
            #========================================================================
            # Prediction
            to_pkl_gzip(obj=pred_result, path=f"../output/pred_result/{start_time}__CV{cvs}__all_preds")
            # Submit File
            pred_result.columns = [COLUMN_ID, COLUMN_TARGET]
            pred_result.iloc[n_rows:].to_csv(f"../submit/tmp/{start_time}__CV{cvs}__feature{n_features}.csv", index=False)
    
    return best_iteration, cv_score, df_feim, pred_result, score_list


def save_log_cv_result(best_iteration, cv_score, model_type, n_features, n_rows, params, score_list, adv_cv_score=-1):
    
    #========================================================================
    # Log
    #========================================================================
    log_map = {}
    log_map['datetime']    = start_time
    log_map['n_features']  = n_features
    log_map['n_rows']      = n_rows
    log_map['cv_score']    = cv_score
    log_map['fold1_score'] = score_list[0]
    log_map['fold2_score'] = score_list[1]
    log_map['fold3_score'] = score_list[2]
    log_map['fold4_score'] = score_list[3]
    log_map['fold5_score'] = score_list[4]
    log_map['best_iteraion']   = int(best_iteration)
    log_map['seed']            = params['seed']
    log_map['metric']          = params['metric']
    log_map['objective']       = params['objective']
    log_map['model_type']      = model_type
    log_map['num_leaves']      = params['num_leaves']    
    log_map['max_depth']       = params['max_depth']     
    log_map['subsample']       = params['subsample']     
    log_map['subsample_freq']  = params['subsample_freq']
    log_map['colsample_bytree']= params['colsample_bytree']
    log_map['lambda_l1']       = params['lambda_l1']     
    log_map['lambda_l2']       = params['lambda_l2']     
    log_map['learning_rate']   = params['learning_rate']     
    log_map['adv_cv_score']    = adv_cv_score
    save_train_log(log_map, params)
    
    
def valid_submit_prediction(prediction):
    """Summary line.
    他Submit Prediction Valueとの相関をチェック
    Args:
    
    Returns:
    """
    list_submit_path = sorted(glob('../submit/validation/*.csv'))
    list_submit = []
    
    print("* Check Corr with Past Submit.")
    for path in list_submit_path:
        lb_score = re.search(rf'([^/LB]*).csv', path).group(1)
        submit = pd.read_csv(path)[COLUMN_TARGET].values
        
        print('  * ', prediction.shape, submit.shape)
        corr = np.min(np.corrcoef(prediction, submit))
        print(f"  * LB{lb_score} / {corr}")



def eval_adversarial_validation(df_train, df_test, COLUMN_GROUP, use_cols, model_type='lgb', params={}):
    #========================================================================
    # Adversarial Validation
    #========================================================================
    df_train['is_train'] = 1
    df_test['is_train']  = 0
    all_data = pd.concat([df_train, df_test], axis=0)
    COLUMN_ADV = 'is_train'
    
    Y_ADV = all_data[COLUMN_ADV]
    all_data.drop(COLUMN_ADV, axis=1, inplace=True)
    kfold = list(GroupKFold(n_splits=5).split(all_data, Y_ADV, all_data[COLUMN_GROUP]))
    
    if len(params)==0:
        params = get_params(model_type)
    
    adv_cv_score, adv_df_feim, adv_pred_result = ieee_cv(
        all_data,
        Y_ADV,
        [],
        use_cols,
        params,
        is_adv=True
    )
    
    return adv_cv_score, adv_df_feim, adv_pred_result


def eval_check_feature(df_train, df_test, is_corr=False):
    # 情報をもたない or 重複してるようなfeatureを除く
    print("* Check Unique Feature.")
    list_unique_drop = drop_unique_feature(df_train, df_test)
    
    if len(list_unique_drop):
        print(f"  * {len(list_unique_drop)}feature unique drop and move trush")
        print(list_unique_drop)
        for col in list(set(list_unique_drop)):
            from_dir = 'valid'
            to_dir = 'valid_trush'
#             if col.count('raw'):
#                 from_dir = 'raw_use'
#                 to_dir = 'raw_trush'
#             else:
#                 from_dir = 'org_use'
#                 to_dir = 'org_trush'
            try:
                move_feature([col], from_dir, to_dir)
            except FileNotFoundError:
                from_dir = 'valid'
                to_dir = 'valid_trush'
                move_feature([col], from_dir, to_dir)
                
    return list_unique_drop

def eval_train(df_train, Y, df_test, COLUMN_GROUP, model_type='lgb', params={}, is_adv=False, is_viz=False, is_valid=False):
    
    use_cols = [col for col in df_train.columns if col not in COLUMNS_IGNORE]
#     df_train = join_same_user(df_train, same_user_path)
#     df_test = join_same_user(df_test, same_user_path)
    
        
    if len(params)==0:
        params = get_params(model_type)
    
    best_iteration, cv_score, df_feim, pred_result, score_list = ieee_cv(
        df_train,
        Y,
        df_test,
        COLUMN_GROUP,
        use_cols,
        params,
        is_valid=is_valid
    )
    
    if is_valid:
        pass
    else:
        test_pred = pred_result.iloc[:, 1].values[len(df_train):]
        valid_submit_prediction(test_pred)
    
    print(f"* CV: {cv_score} | BestIter: {best_iteration}")
    
    if is_viz:
        if model_type=="lgb":
            print("* Training Feature Importance")
            display_importance(df_feim)
    
    #========================================================================
    # Adversarial Validation
    #========================================================================
    if is_adv:
        adv_cv_score, adv_df_feim, adv_pred_result = eval_adversarial_validation(
            df_train,
            df_test,
            COLUMN_GROUP,
            use_cols,
            model_type,
            params,
        )
        print(f"* AdversarialCV: {adv_cv_score}")
    else:
        adv_cv_score = -1
        
    if is_viz and is_adv:
        if model_type=="lgb":
            print("* Adversarial Validation Feature Importance")
            display_importance(adv_df_feim)
        
    #========================================================================
    # 学習結果やパラメータのログをBigQueryに保存する
    #========================================================================
    if is_valid:
        pass
    else:
        n_features = len(use_cols)
        n_rows = df_train.shape[0]
        save_log_cv_result(
            best_iteration,
            cv_score,
            model_type,
            n_features,
            n_rows,
            params,
            score_list,
            adv_cv_score
        )
    
    if is_adv:
        return [df_feim, adv_df_feim]
    else:
        return [df_feim]