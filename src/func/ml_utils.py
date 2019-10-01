from contextlib import contextmanager
from itertools import permutations
from joblib import Parallel, delayed
from time import time, sleep
import datetime
import gc
import glob
import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
import re
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")


import seaborn as sns
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

# Model
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

#========================================================================
# original library 
import func.utils as utils
#========================================================================
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#  import eli5
#  from eli5.sklearn import PermutationImportance

#  perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
#  eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=150)

# CatBoost
#  train_pool = Pool(train_X,train_y)
#  cat_model = CatBoostClassifier(
#                                 iterations=3000,# change 25 to 3000 to get best performance 
#                                 learning_rate=0.03,
#                                 objective="Logloss",
#                                 eval_metric='AUC',
#                                )
#  cat_model.fit(train_X,train_y,silent=True)
#  y_pred_cat = cat_model.predict(X_test)


@contextmanager
def timer(name):
    t0 = time()
    yield
    print(f'''
#========================================================================
# [{name}] done in {time() - t0:.0f} s
#========================================================================
          ''')

#========================================================================
# make feature set
def get_train_test(feat_path_list, base=[], target='target'):
    print(base.shape)
    feature_list = utils.parallel_load_data(path_list=feat_path_list)
    df_feat = pd.concat(feature_list, axis=1)
    df_feat = pd.concat([base, df_feat], axis=1)
    train = df_feat[~df_feat[target].isnull()].reset_index(drop=True)
    test = df_feat[df_feat[target].isnull()].reset_index(drop=True)

    return train, test
#========================================================================


def Regressor(model_type, x_train, x_valid, y_train, y_valid, x_test,
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

        cat_cols = utils.get_categorical_features(df=x_train)

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


def Classifier(
        model_type,
        x_train,
        x_valid,
        y_train,
        y_valid,
        x_test=[],
        params={},
        seed=1208,
        get_model=False,
        get_feim=True,
        early_stopping_rounds=50,
        num_boost_round=3500,
        weight_list=[],
        cols_categorical=[],
):

    if model_type=='lgr':
        params['n_jobs'] = -1
        estimator = LogisticRegression(**params)
    elif model_type=='rmf':
        params['n_jobs'] = -1
        params['n_estimators'] = 10000
        estimator = RandomForestClassifier(**params)
    elif model_type=='cat':
#         params['n_jobs'] = -1
#         params['n_estimators'] = 10000
        estimator = CatBoostClassifier(**params)
    
    elif model_type=='lgb':
        if len(params.keys())==0:
            metric = 'auc'
            params['n_jobs'] = -1
            params['metric'] = metric
            params['num_leaves'] = 31
            params['colsample_bytree'] = 0.3
            params['lambda_l2'] = 1.0
            params['learning_rate'] = 0.1
            params['objective'] = 'binary'

        metric = 'auc'
        params['metric'] = metric
        num_boost_round = num_boost_round

    #========================================================================
    # Fitting
    if model_type!='lgb' and model_type!='cat':
        estimator.fit(x_train, y_train)
    elif model_type=='cat':
        estimator.fit(
            x_train,
            y_train,
            cat_features=cols_categorical,
        )
        best_iter = 1
    else:
        if len(weight_list):
            lgb_train = lgb.Dataset(data=x_train, label=y_train, weight=weight_list[0])
            lgb_valid = lgb.Dataset(data=x_valid, label=y_valid, weight=weight_list[1])
        else:
            lgb_train = lgb.Dataset(data=x_train, label=y_train)
            lgb_valid = lgb.Dataset(data=x_valid, label=y_valid)
            
        estimator = lgb.train(
            params = params,
            train_set = lgb_train,
            valid_sets = lgb_valid,
            early_stopping_rounds = early_stopping_rounds,
            num_boost_round = num_boost_round,
            categorical_feature = cols_categorical,
            verbose_eval = 200
        )
        best_iter = estimator.best_iteration
    #========================================================================

    #========================================================================
    # Prediction
    if model_type=='lgb' or model_type=='cat':
        oof_pred = estimator.predict(x_valid)
        if len(x_test):
            test_pred = estimator.predict(x_test)
        else:
            test_pred = []
    else:
        oof_pred = estimator.predict_proba(x_valid)[:, 1]
        if len(x_test):
            test_pred = estimator.predict_proba(x_test)[:, 1]
        else:
            test_pred = []

#     if metric=='auc':
    score = roc_auc_score(y_valid, oof_pred)
#     elif metric=='logloss':
#         score = log_loss(y_valid, oof_pred)

    if get_feim and model_type=='lgb':
        feim = get_tree_importance(estimator=estimator, use_cols=x_train.columns)
    else:
        feim = []

    if not get_model:
        estimator = None

    return score, oof_pred, test_pred, feim, best_iter, estimator


def get_kfold(train, Y, fold_type='kfold', fold_n=5, seed=1208, shuffle=True):
    if fold_type=='stratified':
        folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    elif fold_type=='kfold':
        folds = KFold(n_splits=fold_n, shuffle=True, random_state=seed)
#     elif fold_type=='group':
#         folds = GroupKFold(n_splits=fold_n, shuffle=True, random_state=seed)

    kfold = list(folds.split(train, Y))

    return kfold


def get_best_blend_ratio(df_pred, y_train, min_ratio=0.01):
    """
    min_ratio: blendを行う比率の最低値
    """

    # 予測値のカラム名
    pred_cols = list(df_pred.columns)

    # blendを行う予測の数
    blend_num = len(pred_cols)

    ratio_list = np.arange(min_ratio, 1.0, min_ratio)
    ratio_combi_list = list(permutations(ratio_list, blend_num))
    ratio_combi_list = [
        combi for combi in ratio_combi_list if np.sum(combi) == 1]

    with timer(f"combi num:{len(ratio_combi_list)}"):

        def get_score_ratio(ratio_combi):

            y_pred_avg = np.zeros(len(df_pred))
            for col, ratio in zip(df_pred.columns, ratio_combi):
                y_pred_avg += df_pred[col].values*ratio
            score = roc_auc_score(y_train, y_pred_avg)
            return [score] + list(ratio_combi)

        score_list = Parallel(
            n_jobs=-1)([delayed(get_score_ratio)(ratio_combi) for ratio_combi in ratio_combi_list])

        df_score = pd.DataFrame(score_list, columns=['score'] + pred_cols)
        df_score.sort_values(by='score', ascending=False, inplace=True)
        best = df_score.iloc[0, :]
        print(f"Best Blend: \n{best}")
        print(df_score.head(10))

        df_corr = pd.DataFrame(np.corrcoef(df_pred.T.values), columns=pred_cols)
        print(f"Corr: \n{df_corr}")

    return best, df_corr


def get_tree_importance(estimator, use_cols, importance_type="gain"):
    feim = estimator.feature_importance(importance_type=importance_type)
    feim = pd.DataFrame([np.array(use_cols), feim]).T
    feim.columns = ['feature', 'importance']
    feim['importance'] = feim['importance'].astype('float32')
    return feim


def display_importance(df_feim, viz_num = 100):
    df_feim = df_feim.reset_index().head(viz_num)
    fig = plt.figure(figsize=(12, 20))
    fig.patch.set_facecolor('white')
    sns.set_style("whitegrid")
    sns.barplot(x="imp_avg", y="feature", data=df_feim)
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgb_importances.png')
    plt.show()


def get_feat_path_list(col_list, file_key='', feat_path='../features/all_features/*.gz'):
    '''
    Explain:
	カラム名の一部が入ったリストを受け取り、featureを検索して
	featureのpathのリストを返す
    Args:
    Return:
    '''

    get_list = []
    path_list = glob.glob(feat_path)

    for path in path_list:
        filename = re.search(r'/([^/.]*).gz', path).group(1)
        for col in col_list:
            if filename.count(col) and filename[:3]==col[:3] and filename.count(file_key):
                get_list.append(path)

    get_list = list(set(get_list))
    return get_list


def split_train_test(df, target):
    train = df[~df[target].isnull()]
    test = df[df[target].isnull()]
    return train, test


def parallel_df(df, func, is_row=False):
    num_partitions = cpu_count()
    num_cores = cpu_count()
    pool = Pool(num_cores)

    if is_raw:
        # 行分割の並列化
        df_split = np.array_split(df, num_partitions) # arrayに変換して行分割する
        df = pd.concat(pool.map(func, df_split))
    else:
        # 列分割の並列化
        df_split = [df[col_name] for col_name in df.columns]
        df = pd.concat(pool.map(func, df_split), axis=1)

    pool.close()
    pool.join()
    return df


#========================================================================
# Relate Feature
#========================================================================

def check_unique_feature(col, feature):
    feature = feature[feature==feature]
    if np.unique(feature[:1000]).shape[0] != 1:
        return True
    if np.unique(feature).shape[0] == 1:
        print(f"{col} has no info.")
        return False
    else:
        return True
    

def drop_unique_feature(df_train, df_test=[]):
    list_drop = []
    for col in tqdm(df_train.columns):
        if df_train[col].value_counts().shape[0] == 1:
            list_drop.append(col)
            continue

    if len(list_drop) > 0:
        print(f'  * {len(list_drop)} No Info Features: {sorted(list_drop)}')
        return list_drop
    else:
        print('All Features have info.')
        return []


def drop_high_corr_feature(df_train, df_test=[], threshold=0.999):
    list_drop = []

    col_corr = set()
    corr_matrix = df_train.corr()
    for i in tqdm(range(len(corr_matrix.columns))):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                col = corr_matrix.columns[i] # getting the name of column
                print(f'highly correlated: {corr_matrix.columns[j]} / {corr_matrix.columns[i]}')
                list_drop.append(col)

    if len(list_drop) > 0:
        return list_drop
    else:
        print('no high correlation columns')
        return []


def get_oof_feature(oof_path='../oof_feature/*.gz', key='', pred_col='prediction'):
    feat_path_list = glob.glob(oof_path)
    oof_list = []
    for path in feat_path_list:
        oof = utils.read_pkl_gzip(path)
        oof_name = oof.columns.tolist()[1]
        oof = oof.set_index(key)[pred_col]
        oof.name = "oof_" + oof_name
        oof_list.append(oof)
    df_oof = pd.concat(oof_list, axis=1)
    return df_oof


def get_label_feature(df, cols_cat):
    for col in cols_cat:
        le = LabelEncoder().fit(df[col])
        df[f"label__{col}"] = le.transform(df[col])
        df.drop(col, axis=1, inplace=True)
    return df


def get_factorize_feature(df, cols_cat, is_sort=True):
    for col in cols_cat:
        df[col], _ = pd.factorize(df[col], sort=is_sort)
    return df


def get_dummie_feature(df, cols_cat, drop=True):

    before_cols = list(df.columns.values)

    for col in cols_cat:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    if drop:
        df.drop(cols_cat, axis=1, inplace=True)

    after_cols = list(df.columns.values)
    dummie_cols = set(after_cols) - set(before_cols)
    for col in dummie_cols:
        df.rename(columns={col:f"{col}_dummie"}, inplace=True)
    return df


def get_cnt_feature(df, columns):
    for col in columns:
        if (str(df[col].dtype) == 'object' or str(df[col].dtype) == 'category' ):
            df[col].fillna('#', inplace=True)
        else:
            df[col].fillna(-98765, inplace=True)
        
        cnt_map = df[col].value_counts().to_dict()
        df[f"cnt__{col}"] = df[col].map(lambda x: cnt_map[x])
        df.drop(col, axis=1, inplace=True)
    return df


def save_feature(df_feat, prefix, dir_save, is_train, auto_type=True, list_ignore=[], is_check=False, is_viz=True):
    
    DIR_FEATURE = Path('../feature') / dir_save
    length = len(df_feat)
    if is_check:
        for col in df_feat.columns:
            if col in list_ignore:
                continue
            # Nullがあるかどうか
            null_len = df_feat[col].dropna().shape[0]
            if length - null_len>0:
                print(f"{col}  | null shape: {length - null_len}")
            # infがあるかどうか
            max_val = df_feat[col].max()
            min_val = df_feat[col].min()
            if max_val==np.inf or min_val==-np.inf:
                print(f"{col} | max: {max_val} | min: {min_val}")
        print("  * Finish Feature Check.")
        sys.exit()
        
    for col in df_feat.columns:
        if col in list_ignore:
            continue
        if auto_type:
            feature = df_feat[col].values.astype('float32')
        else:
            feature = df_feat[col].values
        if is_train:
            feat_path = DIR_FEATURE / f'{prefix}__{col}_train'
        else:
            feat_path = DIR_FEATURE / f'{prefix}__{col}_test'
            
        if os.path.exists(str(feat_path) + '.gz'):
            continue
        else:
            if is_viz:
                print(f"{feature.shape} | {col}")
            utils.to_pkl_gzip(path=str(feat_path), obj=feature)


def replace_inf(df, col):

    feature = df[col].values

    inf_max = np.sort(feature)[::-1][0]
    inf_min = np.sort(feature)[0]

    if inf_max == np.inf:
        for val_max in np.sort(feature)[::-1]:
            if not(val_max==val_max) or val_max==np.inf:
                continue
            feature = np.where(feature==inf_max, val_max, feature)
            break

    if inf_min == -np.inf:
        for val_min in np.sort(feature):
            if not(val_min==val_min) or val_min==-np.inf:
                continue
            feature = np.where(feature==inf_min, val_min, feature)
            break

    length = len(feature)

    #========================================================================
    # infが消えたかチェック
    inf_max = feature.max()
    inf_min = feature.min()
    print(
f"""
#========================================================================
# inf max: {inf_max}
# inf min: {inf_min}
#========================================================================
""")
    #========================================================================

    return feature