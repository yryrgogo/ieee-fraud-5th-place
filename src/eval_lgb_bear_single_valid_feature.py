import gc
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

def get_tree_importance(estimator, use_cols, importance_type="gain"):
    feim = estimator.feature_importance(importance_type=importance_type)
    feim = pd.DataFrame([np.array(use_cols), feim]).T
    feim.columns = ['feature', 'importance']
    feim['importance'] = feim['importance'].astype('float32')
    return feim

def bear_validation(test_pred):
    submission['pred'] = test_pred
    bear_score = submission.merge(bear, how='inner', on=COLUMN_ID)
    public  = bear_score[bear_score['data_type']=='test_public']
    private = bear_score[bear_score['data_type']=='test_private']
    
    public_score = roc_auc_score(public[COLUMN_TARGET].values, public['pred'].values)
    private_score = roc_auc_score(private[COLUMN_TARGET].values, private['pred'].values)
    all_score = roc_auc_score(bear_score[COLUMN_TARGET].values, bear_score['pred'].values)

    return public_score, private_score, all_score


seed = 1208
is_shuffle=False
valid_no = sys.argv[1]
if valid_no=='1':
    is_reverse=False
    i_add = 0
    np.random.seed(seed)
    is_shuffle=True
elif valid_no=='2':
    is_reverse=True
    i_add = 0
    seed = seed*2
    np.random.seed(seed)
    is_shuffle=True
elif valid_no=='3':
    is_reverse=False
    i_add = 12
    seed = seed*3
    np.random.seed(seed)
    is_shuffle=True
elif valid_no=='4':
    is_reverse=True
    i_add = 12
    seed = seed*4
    np.random.seed(seed)
    is_shuffle=True
elif valid_no=='5':
    is_reverse=False
    i_add = 24
    seed = seed*5
    np.random.seed(seed)
    is_shuffle=True
elif valid_no=='6':
    is_reverse=True
    seed = seed*6
    np.random.seed(seed)
    is_shuffle=True
    i_add = 24

valid_paths_train = sorted(glob('../feature/valid/*_train.gz'), reverse=is_reverse)
if is_shuffle:
    valid_paths_train = np.random.choice(valid_paths_train, len(valid_paths_train), False)
else:
    pass
valid_paths_test = [path.replace('train', 'test') for path in valid_paths_train]

save_file_path = '../output/valid_single_feature.csv'
check_score_path = 'check_score.csv'

COLUMN_ID = 'TransactionID'
COLUMN_DT = 'TransactionDT'
COLUMN_TARGET = 'isFraud'
COLUMN_GROUP = 'DT-M'
COLUMNS_IGNORE = [COLUMN_ID, COLUMN_DT, COLUMN_TARGET, COLUMN_GROUP, 'is_train', 'date']

paths_train = glob('../feature/raw_use/*_train.gz')
paths_train += sorted(glob('../feature/org_use/*_train.gz'))

paths_test = glob('../feature/raw_use/*_test.gz')
paths_test += sorted(glob('../feature/org_use/*_test.gz'))

df_train = parallel_load_data(paths_train)
df_test = parallel_load_data(paths_test)

group_kfold_path = '../input/0908_ieee__DT-M_GroupKFold.gz'
group = read_pkl_gzip(group_kfold_path)
df_train[COLUMN_GROUP] = group


#========================================================================
# Negative Down Sampling
#========================================================================
frac = 0.2
np.random.seed(seed)
df_pos = df_train[df_train.isFraud==1]
df_neg = df_train[df_train.isFraud!=1]
del df_train
gc.collect()
df_neg = df_neg.sample(int(df_neg.shape[0] * frac))
df_train = pd.concat([df_pos, df_neg], axis=0)

# Cut 2017-12
df_train = df_train[df_train[COLUMN_GROUP]!='2017-12']

#========================================================================
# Base Featureに検証用Feature Groupを追加して、スコアの変化を見る.
# Baseより向上したFeature Groupのみ、追加検証を行う
#========================================================================

# is_baseをTrueにして基準を記録する
is_base = [True, False][1]
is_result = [True, False][1]
is_write  = [True, False][0]

fold_map = {
    0: '2018-5',
    1: '2018-4',
    2: '2018-3',
}
base_fold_score = {
    0: 0.93600,
    1: 0.94750,
    2: 0.93750,
}

print("Num Feature", len(valid_paths_train))

for i in range(8):

    start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())[:14]
    #  valid_path = valid_paths_train[(i+i_add):(i+i_add)+1]
    valid_path = valid_paths_train[(i+i_add):(i+i_add+1)]
    valid_path_test = valid_paths_test[(i+i_add):(i+i_add+1)]

    list_done = pd.read_csv('done.csv').values
    if valid_path[0] in list_done:
        print('Done Feature.')
        continue

    with open('done.csv', 'a') as f:
        line = f'{valid_path[0]}\n'
        f.write(line)
    
    # 既に他のプロセスが実行済みだったらスキップ
    if os.path.exists(valid_path[0]):
        pass
    else:
        print('No exist path.')
        continue
    
    
    if is_base or len(valid_path)==0:
        tmp_train = df_train.copy()
        feature_name = 'base'
    else:
        df_feat_train = parallel_load_data(valid_path)
        tmp_train = df_train.join(df_feat_train)
        
        df_feat_test = parallel_load_data(valid_path_test)
        tmp_test = df_test.join(df_feat_test)
        
        feature_name = get_filename(valid_path[0])
    
    use_cols = [col for col in tmp_train.columns if col not in COLUMNS_IGNORE]
    feim_list = []
    x_test = tmp_test[use_cols]
    
    cnt = 0    
    cv = 0
    for fold in range(3):
        with timer('  * Make Dataset'):
            if fold==0:
                train = tmp_train[
#                     (tmp_train[COLUMN_GROUP] == '2017-12') | 
                    (tmp_train[COLUMN_GROUP] == '2018-1') | 
                    (tmp_train[COLUMN_GROUP] == '2018-2') | 
                    (tmp_train[COLUMN_GROUP] == '2018-3') | 
                    (tmp_train[COLUMN_GROUP] == '2018-4')
                    ]
                test  = tmp_train[tmp_train[COLUMN_GROUP] == '2018-5']
            elif fold==1:
                train = tmp_train[
#                     (tmp_train[COLUMN_GROUP] == '2017-12') | 
                    (tmp_train[COLUMN_GROUP] == '2018-1') | 
                    (tmp_train[COLUMN_GROUP] == '2018-2') | 
                    (tmp_train[COLUMN_GROUP] == '2018-3') |
                    (tmp_train[COLUMN_GROUP] == '2018-5')
                    ]
                test  = tmp_train[tmp_train[COLUMN_GROUP] == '2018-4']
            elif fold==2:
                train = tmp_train[
#                     (tmp_train[COLUMN_GROUP] == '2017-12') | 
                    (tmp_train[COLUMN_GROUP] == '2018-1') | 
                    (tmp_train[COLUMN_GROUP] == '2018-2') | 
                    (tmp_train[COLUMN_GROUP] == '2018-4') |
                    (tmp_train[COLUMN_GROUP] == '2018-5')
                    ]
                test  = tmp_train[tmp_train[COLUMN_GROUP] == '2018-3']
        
            Y_TRAIN = train[COLUMN_TARGET]
            train.drop(COLUMN_TARGET, axis=1, inplace=True)
        
            Y_TEST = test[COLUMN_TARGET]
            test.drop(COLUMN_TARGET, axis=1, inplace=True)
        
        params = {
#             'n_jobs': 64,
            'n_jobs': 16,
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
        num_boost_round=500
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
            
            feim = get_tree_importance(estimator, use_cols)
            feim.rename(columns={'importance': f'imp_fold{fold+1}'}, inplace=True)
            feim.set_index('feature', inplace=True)
            feim_list.append(feim)
            df_feim = pd.concat(feim_list, axis=1)
            df_feim['imp_avg'] = df_feim.mean(axis=1)
            df_feim.sort_values(by='imp_avg', ascending=False, inplace=True)
            
            to_pkl_gzip(obj=df_feim, path=f"../output/feature_importances/{start_time}__bear_valid__CV{cvs}__feature{len(use_cols)}")
            
            test_pred = estimator.predict(x_test)
            pb, pv, al = bear_validation(test_pred)

            logger.info(f"  * {feature_name} Fold{fold} {fold_map[fold]}:{score} | Bear's PB: {pb} PV: {pv} ALL: {al}")
                    
            if not is_result and is_write:
                with open(save_file_path, 'a') as f:
                    line = f'{start_time},{fold_map[fold]},{feature_name},{score},{pb},{pv},{al}\n'
                    f.write(line)

#             # 三行もたないfeatureは各foldをクリアできなかった
#             if score < base_fold_score[fold]:
#                 break
#             else:
#                 cnt +=1
#                 cv += score/3

            
    if cnt==3:
        with open(check_score_path, 'a') as f:
            line = f'{feature_name},{cv}\n'
            f.write(line)
            
        df_score = pd.read_csv(check_score_path, header=None)
        if len(df_score)>2:
            from_dir = 'valid'
            to_dir = 'sub_use'
            df_score.columns = ['feature', 'score']
            df_score.sort_values(by='score', ascending=False, inplace=True)
            best_feature = df_score['feature'].values[0]
            if best_feature.count('_train'):
                best_feature = best_feature.replace('_train', '')
            move_feature([best_feature], from_dir, to_dir)
            os.system(f'rm {check_score_path}')
            os.system(f'touch {check_score_path}')
            
    
    #========================================================================
    # PostProcess
    #========================================================================
    to_dir = '../feature/check_trush/'
    with timer("  * PostProcess"):
        for path in valid_path:
            try:
                shutil.move(path, to_dir)
                shutil.move(path.replace('_train', '_test'), to_dir)
            except FileNotFoundError:
                print(feature_name)
