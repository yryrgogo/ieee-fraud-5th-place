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
import warnings
warnings.filterwarnings('ignore')

valid_no = int(sys.argv[1])

early_stopping_rounds = 50
COLUMN_ID = 'TransactionID'
COLUMN_DT = 'TransactionDT'
COLUMN_GROUP = 'DT-M'
COLUMN_TARGET = 'isFraud'
COLUMNS_IGNORE = [COLUMN_ID, COLUMN_DT, COLUMN_TARGET, 'is_train', 'date', COLUMN_GROUP]

paths_train = glob('../feature/raw_use/*_train.gz')
paths_train = [path for path in paths_train]
paths_test = glob('../feature/raw_use/*_test.gz')
paths_test = [path for path in paths_test]
paths_train += sorted(glob('../feature/org_use/*_train.gz'))
paths_test  += sorted(glob('../feature/org_use/*_test.gz'))


list_feim = []
valid_paths_train = sorted(glob('../feature/valid/*_train.gz'))[:500]
valid_paths_test  = sorted(glob('../feature/valid/*_test.gz'))[:500]

if len(valid_paths_train)==0:
    sys.exit()

#  df_train = reduce_mem_usage( parallel_load_data(paths_train) )
#  df_test  = reduce_mem_usage( parallel_load_data(paths_test) )
df_train = parallel_load_data(paths_train)
df_test  = parallel_load_data(paths_test)
Y = df_train[COLUMN_TARGET]
df_train.drop(COLUMN_TARGET, axis=1, inplace=True)

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

df_feat_train =  parallel_load_data(valid_paths_train)
df_feat_test  =  parallel_load_data(valid_paths_test)

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


#========================================================================
# GroupKFold
#========================================================================
group_kfold_path = '../input/0908_ieee__DT-M_GroupKFold.gz'
group = read_pkl_gzip(group_kfold_path)
tmp_train[COLUMN_GROUP] = group
# same_user_path = '../output/same_user_pattern/20190901_user_ids_share.csv'
#  same_user_path = '../output/same_user_pattern/0902__same_user_id__card_addr_pemail_M.csv'


model_type = "lgb"
params = {
   'n_jobs': 64,
    #  'n_jobs': 48,
    'seed': 1208,
    'n_splits': 6,
    'metric': 'auc',
    'model_type': model_type,
    'objective': 'binary',
    'fold': ['stratified', 'group'][1],
    'num_leaves': 2**8-1,
    'max_depth': 8,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree' : 0.20,
    'lambda_l1' : 0.1,
    'lambda_l2' : 1.0,
    'learning_rate' : 0.1,
    'early_stopping_rounds' : early_stopping_rounds,
}
list_result_feim = eval_train(
    tmp_train,
    Y,
    tmp_test,
    COLUMN_GROUP,
    model_type,
    params,
    is_adv=[True, False][1],
    is_viz=[True, False][1],
    is_valid=True,
)

#========================================================================
# Importance Gain が最大のfeatureの1%未満のGainしかもたないFeatureを除外する
#========================================================================

feim = list_result_feim[0]
max_imp = feim['imp_avg'].max()
#  valid_features = [i for i in feim.index if i.startswith('503') or i.startswith('504') or i.startswith('505')]
thres_imp = 500
#  thres_imp = max_imp/50
#  for feature_name in feim[feim['imp_avg']<thres_imp].index:
for feature_name in feim.iloc[550:].index:
    from_dir = 'valid'
    to_dir = 'valid_trush'
    try:
        move_feature([feature_name], from_dir, to_dir)
    except FileNotFoundError:
        print(feature_name)


for feature_name in feim.iloc[:550].index:
#  for feature_name in valid_features[100:]:
    from_dir = 'valid'
    to_dir = 'valid_use'
    try:
        move_feature([feature_name], from_dir, to_dir)
    except FileNotFoundError:
        pass
