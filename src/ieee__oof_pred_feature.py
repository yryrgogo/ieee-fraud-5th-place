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
from func.utils import get_categorical_features, read_pkl_gzip, to_pkl_gzip, parallel_load_data, get_filename, logger_func, timer
from ieee_train import eval_train, eval_check_feature
from kaggle_utils import reduce_mem_usage, move_feature
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, r2_score, mean_squared_error
import lightgbm as lgb 

try:
    logger
except NameError:
    logger = logger_func()
    
    
COLUMN_TARGET = sys.argv[1]

COLUMN_ID = 'TransactionID'
COLUMN_DT = 'TransactionDT'
COLUMN_GROUP = 'DT-M'
COLUMNS_IGNORE = [COLUMN_ID, COLUMN_DT, COLUMN_TARGET, COLUMN_GROUP, 'isFraud', 'is_train', 'date', 'DT-M', 'predicted_user_id']


def get_tree_importance(estimator, use_cols, importance_type="gain"):
    feim = estimator.feature_importance(importance_type=importance_type)
    feim = pd.DataFrame([np.array(use_cols), feim]).T
    feim.columns = ['feature', 'importance']
    feim['importance'] = feim['importance'].astype('float32')
    return feim


def Regressor(base_valid, model_type, x_train, x_valid, y_train, y_valid, x_test,
    params={}, seed=1208, get_score='rmse', get_model=False,
    early_stopping_rounds=100, num_boost_round=10000):

    if str(type(x_train)).count('DataFrame'):
        use_cols = x_train.columns
    else:
        use_cols = np.arange(x_train.shape[1]) + 1

    if model_type=='linear':
        estimator = LinearRegression(**params)
    elif model_type=='ridge':
        estimator = Ridge(**params)
    elif model_type=='lasso':
        estimator = Lasso(**params)
    elif model_type=='rmf':
        params['n_jobs'] = -1
        params['n_estimators'] = 10000
        estimator = RandomForestRegressor(**params)
    elif model_type=='lgb':
        if len(params.keys())==0:
            metric = 'auc'
            params['n_jobs'] = 32
            params['metric'] = metric
            params['num_leaves'] = 31
            params['colsample_bytree'] = 0.3
            params['lambda_l2'] = 1.0
            params['learning_rate'] = 0.01
        num_boost_round = num_boost_round
        params['objective'] = 'regression'
        params['metric'] = 'mse'

    #========================================================================
    # Fitting
    if model_type!='lgb':
        estimator.fit(x_train, y_train)
    else:
        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        lgb_valid = lgb.Dataset(data=x_valid, label=y_valid)

        cat_cols = get_categorical_features(df=x_train)

        estimator = lgb.train(
            params = params
            ,train_set = lgb_train
            ,valid_sets = lgb_valid
            ,early_stopping_rounds = early_stopping_rounds
            ,num_boost_round = num_boost_round
            ,categorical_feature = cat_cols
            ,verbose_eval = 200
        )

    #========================================================================

    #========================================================================
    # Prediction
    oof_pred = estimator.predict(x_valid)
    if len(x_test):
        test_pred = estimator.predict(x_test)
    else:
        test_pred = []
    #========================================================================

    #========================================================================
    # Scoring
    score = np.sqrt(mean_squared_error(y_valid, oof_pred))
    oof_pred = estimator.predict(base_valid)
    # Model   : {model_type}
    # feature : {x_train.shape, x_valid.shape}
    #========================================================================

    if model_type=='lgb':
        feim = get_tree_importance(estimator=estimator, use_cols=x_train.columns)
        feim.sort_values(by='importance', ascending=False, inplace=True)
    elif model_type=='lasso' or model_type=='ridge':
        feim = pd.Series(estimator.coef_, index=use_cols, name='coef')
        feim.sort_values(ascending=False, inplace=True)

    if get_model:
        return score, oof_pred, test_pred, feim, estimator
    else:
        return score, oof_pred, test_pred, feim, 0


def filter_feature(path):
    if path.count(''):
        return True
    else:
        return False

paths_train = glob('../submit/re_sub/*_train.gz')
paths_test  = glob('../submit/re_sub/*_test.gz')
# paths_train += glob('../submit/re_sub/Tran*_train.gz')
# paths_test  += glob('../submit/re_sub/Tran*_test.gz')
# paths_train += glob('../submit/re_sub/is*_train.gz')
# paths_test  += glob('../submit/re_sub/is*_test.gz')

print(len(paths_train))

df_train = parallel_load_data(paths_train)
df_test  = parallel_load_data(paths_test)

### DT-M
group_kfold_path = '../input/0908_ieee__DT-M_GroupKFold.gz'
group = read_pkl_gzip(group_kfold_path)
df_train[COLUMN_GROUP] = group

Y = df_train[COLUMN_TARGET]


is_submit = [True, False][0]
n_splits = 6
set_type = 'new_set'

tmp_train = df_train
tmp_test = df_test

#========================================================================
# Train Test で片方に存在しないFeatureを除外
#========================================================================
diff_cols = list(set(tmp_train.columns) - set(tmp_test.columns) - set(COLUMNS_IGNORE) - set([COLUMN_TARGET]))

for col in list(set(diff_cols)):
    from_dir = 'valid'
    to_dir = 'valid_trush'
    move_feature([col], from_dir, to_dir)
tmp_train.drop(diff_cols, axis=1, inplace=True)
print(f"  * Diff Features: {len(diff_cols)}")


model_type = "lgb"
params = {
#     'n_jobs': 60,
    'n_jobs': 96,
#     'n_jobs': 84,
#     'n_jobs': 48,
#     'n_jobs': 36,
    'objective': 'regression',
    'num_leaves': 2**8-1,
    'max_depth': -1,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree' : 0.10,
    'lambda_l1' : 0.1,
    'lambda_l2' : 1.0,
    'learning_rate' : 0.1,
    "early_stopping_rounds": 50,
    "seed": 1208,
    "bagging_seed": 1208,
    "feature_fraction_seed": 1208,
    "drop_seed": 1208,
    'n_splits': n_splits,
    'metric': 'auc',
    'model_type': model_type,
    'fold': ['stratified', 'group'][1],
}
if is_submit:
    params['learning_rate'] = 0.01
    params['learning_rate'] = 0.05

logger.info(f"* EXP: dataset {set_type} {tmp_train.shape} lr {params['learning_rate']} ")
            
            
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())[:13]
seed       = params['seed']
model_type = params['model_type']
n_splits = params['n_splits']
validation = params['fold']
early_stopping_rounds = params['early_stopping_rounds']

use_cols = [col for col in tmp_train.columns if col not in COLUMNS_IGNORE]

kfold = list(GroupKFold(n_splits=n_splits).split(tmp_train, Y, tmp_train[COLUMN_GROUP]))

score_list = []
feim_list = []
y_pred = np.zeros(len(tmp_train))
test_preds = []

x_test = df_test[use_cols]

for n_fold, (trn_idx, val_idx) in enumerate(kfold):
    x_train = tmp_train.iloc[trn_idx]
    y_train = Y.iloc[trn_idx]
    x_valid = tmp_train.iloc[val_idx]
    y_valid = Y.iloc[val_idx]
            
    x_train = x_train[~x_train[COLUMN_TARGET].isnull()][use_cols]
    x_trn_idx = x_train.index
    x_valid = x_valid[~x_valid[COLUMN_TARGET].isnull()][use_cols]
    x_val_idx = x_valid.index
    y_train = y_train.loc[x_trn_idx]
    y_valid = y_valid.loc[x_val_idx]
            
    base_valid = tmp_train.iloc[val_idx][use_cols]

    val_gr = tmp_train.iloc[val_idx][COLUMN_GROUP].value_counts()
    dtm = val_gr.index.tolist()[0]
    print("="*20)
    with timer(f"  * Fold{n_fold} Validation-{COLUMN_GROUP} {dtm}: {val_gr.values[0]}"):
        score, oof_pred, test_pred, feim, _ = Regressor(
            base_valid,
            model_type=model_type,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            x_test=x_test,
            params=params,
            early_stopping_rounds = early_stopping_rounds,
        )

    score_list.append(score)
    y_pred[val_idx] = oof_pred
    test_preds.append(test_pred)

    feim.rename(columns={'importance': f'imp_fold{n_fold+1}'}, inplace=True)
    feim.set_index('feature', inplace=True)
    feim_list.append(feim)

cv_score = np.mean(score_list)
cvs = str(cv_score).replace('.', '-')
df_feim = pd.concat(feim_list, axis=1)
df_feim['imp_avg'] = df_feim.mean(axis=1)
df_feim.sort_values(by='imp_avg', ascending=False, inplace=True)

## Save
# Feature Importance
to_pkl_gzip(obj=df_feim, path=f"../output/feature_importances/{start_time}__CV{cvs}__{COLUMN_TARGET}__feature{len(use_cols)}")


with timer("  * Make Prediction Result File."):
    test_pred_avg = np.mean(test_preds, axis=0)
    all_pred = np.append(y_pred, test_pred_avg)
    all_ids = np.append(tmp_train[COLUMN_ID].values, df_test[COLUMN_ID].values)
    pred_result = pd.DataFrame([all_ids, all_pred], index=[COLUMN_ID, 'pred_' + start_time]).T
    pred_result[COLUMN_ID] = pred_result[COLUMN_ID].astype('int')

    #========================================================================
    # Save
    #========================================================================
    # Prediction
    to_pkl_gzip(obj=pred_result, path=f"../output/pred_result/{start_time}__CV{cvs}__{COLUMN_TARGET}__all_preds")
    # Submit File
#     pred_result.columns = [COLUMN_ID, COLUMN_TARGET]
#     pred_result.iloc[len(tmp_train):].to_csv(f"../submit/tmp/{start_time}__CV{cvs}__feature{len(use_cols)}.csv", index=False)
