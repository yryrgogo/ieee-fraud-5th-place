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
from func.utils import get_categorical_features, read_pkl_gzip, to_pkl_gzip, parallel_load_data, get_filename
from ieee_train import eval_train, eval_check_feature
from kaggle_utils import reduce_mem_usage, move_feature


COLUMN_ID = 'TransactionID'
COLUMN_DT = 'TransactionDT'
COLUMN_TARGET = 'isFraud'
COLUMNS_IGNORE = [COLUMN_ID, COLUMN_DT, COLUMN_TARGET, 'is_train', 'date']

paths_train = glob('../feature/raw_use/*_train.gz')
paths_train = [path for path in paths_train 
               if path.count('Fraud') 
               or path.count('tionID') 
               or path.count('C1')
               or path.count('C13')
               or path.count('V232')
               or path.count('addr1')
]
paths_test = glob('../feature/raw_use/*_test.gz')
paths_test = [path for path in paths_test 
               if path.count('Fraud') 
               or path.count('tionID') 
               or path.count('C1')
               or path.count('C13')
               or path.count('V232')
               or path.count('addr1')
]
paths_train_feature = sorted(glob('../feature/org_use/*_train.gz'))
paths_test_feature  = sorted(glob('../feature/org_use/*_test.gz'))

df_train = reduce_mem_usage( parallel_load_data(paths_train) )
df_test  = reduce_mem_usage( parallel_load_data(paths_test) )
Y = df_train[COLUMN_TARGET]
df_train.drop(COLUMN_TARGET, axis=1, inplace=True)

valid_no = int(sys.argv[1])

list_feim = []
valid_paths_train = paths_train_feature[valid_no * 800 : (valid_no + 1) * 800]
valid_paths_test  = paths_test_feature[valid_no * 800  : (valid_no + 1) * 800]

#========================================================================
# pathの存在チェック。なぜかたびたびFileNotFoundErrorが起きるので,,,
#========================================================================
remove_paths = []
for trn_path, tes_path in zip(valid_paths_train, valid_paths_test):
    if os.path.exists(trn_path) and os.path.exists(tes_path):
        pass
    else:
        remove_paths.append(trn_path)
        remove_paths.append(tes_path)
for path in remove_paths:
    if path.count('train'):
        valid_paths_train.remove(path)
        print(f'remove {path}')
    elif path.count('test'):
        valid_paths_test.remove(path)
        print(f'remove {path}')

df_feat_train = reduce_mem_usage( parallel_load_data(valid_paths_train) )
df_feat_test  = reduce_mem_usage( parallel_load_data(valid_paths_test) )

col_drops = eval_check_feature(df_feat_train, df_feat_test)

tmp_train = df_train.join(df_feat_train)
tmp_test = df_test.join(df_feat_test)

#========================================================================
# Train Test で片方に存在しないFeatureを除外
#========================================================================
diff_cols = list(set(tmp_train.columns) - set(tmp_test.columns))
for col in list(set(diff_cols)):
    if col.count('raw'):
        from_dir = 'raw_use'
        to_dir = 'raw_trush'
    else:
        from_dir = 'org_use'
        to_dir = 'org_trush'
    move_feature([col], from_dir, to_dir)
tmp_train.drop(diff_cols, axis=1, inplace=True)

# same_user_path = '../output/same_user_pattern/20190901_user_ids_share.csv'
same_user_path = '../output/same_user_pattern/0902__same_user_id__card_addr_pemail_M.csv'
model_type = "lgb"
params = {
    'n_jobs': 60,
    'seed': 1208,
    'n_splits': 5,
    'metric': 'auc',
    'model_type': model_type,
    'objective': 'binary',
    'fold': ['stratified', 'group'][1],
#     'num_leaves': 2**6-1,
    'num_leaves': 2**5-1,
    'max_depth': 8,
    'subsample': 0.75,
    'subsample_freq': 1,
#     'colsample_bytree' : 0.20,
    'colsample_bytree' : 0.15,
    'lambda_l1' : 0.1,
    'lambda_l2' : 1.0,
    'learning_rate' : 0.1,
}
list_result_feim = eval_train(
    tmp_train,
    Y,
    tmp_test,
    same_user_path,
    model_type,
    params,
    is_adv=[True, False][1],
    is_viz=[True, False][1],
)
list_feim.append(list_result_feim)

#========================================================================
# Importance Gain が最大のfeatureの1%未満のGainしかもたないFeatureを除外する
#========================================================================

feim = list_result_feim[0]
max_imp = feim['imp_avg'].max()
# thres_imp = max_imp/100
#  thres_imp = max_imp/25
#  for feature_name in feim[feim['imp_avg']<thres_imp].index:
for feature_name in feim.tail(feim.shape[0]-50).index:
    if feature_name.count('raw'):
        from_dir = 'raw_use'
        to_dir = 'raw_trush'
    else:
        from_dir = 'org_use'
        to_dir = 'org_trush'
    try:
        move_feature([feature_name], from_dir, to_dir)
    except FileNotFoundError:
        print(feature_name)
