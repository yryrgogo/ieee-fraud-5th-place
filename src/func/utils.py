"""
gsutil -m cp ../feature/* gs://homecredit_ko
gsutil -m rsync -d -r ../feature gs://homecredit_ko
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import re
import os
import subprocess
import sys
from socket import gethostname
HOSTNAME = gethostname()

from tqdm import tqdm
from sklearn.model_selection import KFold
import time
from datetime import datetime
import gc
import pickle
import gzip
from multiprocessing import Pool
import multiprocessing
from contextlib import contextmanager

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.model_selection import StratifiedKFold, KFold

# =============================================================================
# global variables
# =============================================================================
COMPETITION_NAME = 'home-credit-default-risk'
#  COMPETITION_NAME = 'elo-merchant-category-recommendation'
#  COMPETITION_NAME = 'microsoft-malware-prediction'

# =============================================================================
# def
# =============================================================================


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def start(fname):
    global st_time
    st_time = time()
    print(f"""
#==============================================================================
# START!!! {fname}    PID: {os.getpid()}    time: {datetime.today()}
#==============================================================================
""")
    send_line(f'{HOSTNAME}  START {fname}  time: {elapsed_minute():.2f}min')
    return


def reset_time():
    global st_time
    st_time = time()
    return


def end(fname):
    print(f"""
#==============================================================================
# SUCCESS !!! {fname}
#==============================================================================
""")
    print('time: {:.2f}min'.format(elapsed_minute()))
    send_line(f'{HOSTNAME}  FINISH {fname}  time: {elapsed_minute():.2f}min')
    return


def read_text(file_path):
    with open(file_path) as f:
        text = f.read()
        #  text_list = text.split()
    return text


def elapsed_minute():
    return (time() - st_time)/60


def to_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj=obj, file=f)
        print(f"""
#==============================================================================
# PICKLE TO SUCCESS !!! {path}
#==============================================================================
""")


def read_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        print(f"""
#==============================================================================
# PICKLE READ SUCCESS !!! {path}
#==============================================================================
""")
        return obj


def to_pkl_gzip(obj, path):
    #  df.to_pickle(path)
    with open(path, 'wb') as f:
        pickle.dump(obj=obj, file=f)
    os.system('gzip -f ' + '''"'''+path+'''"''')
    os.system('rm ' + '''"'''+path+'''"''')
    return


def read_pkl_gzip(path):
    with gzip.open(path, mode='rb') as fp:
        data = fp.read()
    return pickle.loads(data)


def to_df_pkl(df, path, fname='', split_size=3, index=False):
    """
    path = '../output/mydf'

    write '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'

    """
    print(f'shape: {df.shape}')

    if index:
        df.reset_index(drop=True, inplace=True)
        gc.collect()
    mkdir_func(path)

    ' データセットを切り分けて保存しておく '
    kf = KFold(n_splits=split_size)
    #  for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
    for i, (train_index, val_index) in enumerate(kf.split(df)):
        df.iloc[val_index].to_pickle(f'{path}/{fname}{i:03d}.p')
    return


def read_df_pkl(path, col=None, use_tqdm=True):
    if col is None:
        if use_tqdm:
            df = pd.concat([pd.read_pickle(f)
                            for f in tqdm(sorted(glob(path)))])
        else:
            print(f'reading {path}')
            df = pd.concat([pd.read_pickle(f)
                            for f in sorted(glob(path))])
    else:
        df = pd.concat([pd.read_pickle(f)[col]
                        for f in tqdm(sorted(glob(path)))])
    return df


def merge(df, col):
    trte = pd.concat([load_train(col=col),  # .drop('TARGET', axis=1),
                      load_test(col=col)])
    df_ = pd.merge(df, trte, on='SK_ID_CURR', how='left')
    return df_


def check_feature():

    sw = False
    files = sorted(glob('../feature/train*.f'))
    for f in files:
        path = f.replace('train_', 'test_')
        if not os.path.isfile(path):
            print(f)
            sw = True

    files = sorted(glob('../feature/test*.f'))
    for f in files:
        path = f.replace('test_', 'train_')
        if not os.path.isfile(path):
            print(f)
            sw = True

    if sw:
        raise Exception('Miising file :(')
    else:
        print('All files exist :)')

# =============================================================================
#
# =============================================================================

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def check_var(df, var_limit=0, sample_size=None):
    if sample_size is not None:
        if df.shape[0] > sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            df_ = df
#            raise Exception(f'df:{df.shape[0]} <= sample_size:{sample_size}')
    else:
        df_ = df

    var = df_.var()
    col_var0 = var[var <= var_limit].index
    if len(col_var0) > 0:
        print(f'remove var<={var_limit}: {col_var0}')
    return col_var0


def check_corr(df, corr_limit=1, sample_size=None):
    if sample_size is not None:
        if df.shape[0] > sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            raise Exception(f'df:{df.shape[0]} <= sample_size:{sample_size}')
    else:
        df_ = df

    corr = df_.corr('pearson').abs()  # pearson or spearman
    a, b = np.where(corr >= corr_limit)
    col_corr1 = []
    for a_, b_ in zip(a, b):
        if a_ != b_ and a_ not in col_corr1:
            #            print(a_, b_)
            col_corr1.append(b_)
    if len(col_corr1) > 0:
        col_corr1 = df.iloc[:, col_corr1].columns
        print(f'remove corr>={corr_limit}: {col_corr1}')
    return col_corr1


def remove_feature(df, var_limit=0, corr_limit=1, sample_size=None, only_var=True):
    col_var0 = check_var(df,  var_limit=var_limit, sample_size=sample_size)
    df.drop(col_var0, axis=1, inplace=True)
    if only_var == False:
        col_corr1 = check_corr(df, corr_limit=corr_limit,
                               sample_size=sample_size)
        df.drop(col_corr1, axis=1, inplace=True)
    return


def __get_use_files__():

    return


# =============================================================================
# other API
# =============================================================================
def submit(file_path, comment='from API', COMPETITION_NAME=''):
    os.system(f'kaggle competitions submit -c {COMPETITION_NAME} -f {file_path} -m "{comment}"')
    time.sleep(30)  # tekito~~~~
    tmp = os.popen(f'kaggle competitions submissions -c {COMPETITION_NAME} -v | head -n 2').read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i, j in zip(col.split(','), values.split(',')):
        message += f'{i}: {j}\n'
        print(f'{i}: {j}') # TODO: comment out later?
        if i.count('public'):
            lb = j
        elif i.count('private'):
            pb = j
    return [lb, pb]
    #  send_line(message.rstrip())


import requests


def send_line(message):

    line_notify_token = '5p5sPTY7PrQaB8Wnwp6aadfiqC8m2zh6Q8llrfNisGT'
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)


def stop_instance():
    """
    You need to login first.
    >> gcloud auth login
    """
    send_line('stop instance')
    os.system(
        f'gcloud compute instances stop {os.uname()[1]} --zone us-east1-b')


def mkdir_func(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

" 機械学習でよく使う系 "
def logger_func():
    logger = getLogger(__name__)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s]\
    [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    mkdir_func('../output')
    handler = FileHandler('../output/py_train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    return logger


def load_file(path, delimiter='gz'):
    if path.count('.csv'):
        return pd.read_csv(path)
    filename = get_filename(path=path, delimiter=delimiter)

    if filename.count('train_'):
        filename = filename[6:]
    elif filename.count('test_'):
        filename = filename[5:]

    if path[-3:]=='npy':
        tmp = pd.Series(np.load(path), name=filename)
    elif path[-2:]=='fp':
        with gzip.open(path, mode='rb') as fp:
            data = fp.read()
            tmp = pd.Series(pickle.loads(data), name=filename)
    elif path[-2:]=='gz':
        with gzip.open(path, mode='rb') as gz:
            data = gz.read()
            tmp = pd.Series(pickle.loads(data), name=filename)
    return tmp

def get_filename(path, delimiter='gz'):
    if path.count(r'.'):
        path = path.replace(r'.', '_')
    filename = re.search(rf'/([^/.]*).{delimiter}', path).group(1)
    return filename

def parallel_load_data(path_list, feature_names, n_jobs=multiprocessing.cpu_count()):
    p = Pool(n_jobs)
    p_list = p.map(load_file, path_list)
    p.close
    result = pd.DataFrame(p_list, index=feature_names).T
    return result

def load_file_wrapper(args):
    return load_file(*args)


def parallel_process(func, arg_list, cpu_cnt=multiprocessing.cpu_count()):
    process = Pool(cpu_cnt)
    #  p = Pool(len(arg_list))
    callback = process.map(func, arg_list)
    process.close()
    process.terminate()
    return callback


def row_number(df, level):
    '''
    Explain:
        levelをpartisionとしてrow_numberをつける。
        順番は入力されたDFのままで行う、ソートが必要な場合は事前に。
    Args:
    Return:
    '''
    index = np.arange(1, len(df)+1, 1)
    df['index'] = index
    min_index = df.groupby(level)['index'].min().reset_index()
    df = df.merge(min_index, on=level, how='inner')
    df['row_no'] = df['index_x'] - df['index_y'] + 1
    df.drop(['index_x', 'index_y'], axis=1, inplace=True)
    return df


#  カテゴリ変数を取得する関数
def get_categorical_features(df, ignore_list=[]):
    obj = [col for col in list(df.columns) if (str(df[col].dtype) == 'object' or str(df[col].dtype) == 'category' ) and col not in ignore_list]
    return obj

#  カテゴリ変数を取得する関数
def get_datetime_features(df, ignore_list=[]):
    dt = [col for col in list(df.columns) if str(df[col].dtype).count('time') and col not in ignore_list]
    return dt

#  連続値カラムを取得する関数
def get_numeric_features(df, ignore_list=[]):
    num = [col for col in list(df.columns) if (str(df[col].dtype).count('int') or str(df[col].dtype).count('float')) and col not in ignore_list]
    return num


def round_size(value, max_val, min_val):
    range_val = int(max_val - min_val)
    origin_size = len(str(value))
    dtype = str(type(value))
    if dtype.count('float') or dtype.count('decimal'):
        origin_num = str(value)
        str_num = str(int(value))
        int_size = len(str_num)
        d_size = origin_size - int_size -1
        if int_size>1:
            if origin_num[-1]=='0':
                if range_val>10:
                    dec_size=0
                else:
                    dec_size=2
            else:
                dec_size = 1
        else:
            if d_size>3:
                dec_size = 3
            else:
                dec_size = 2
    elif dtype.count('int'):
            dec_size = 0
    else:
        raise ValueError('date type is int or float or decimal.')

    return dec_size



#========================================================================
# Kaggle Elo
#========================================================================
def save_feature(prefix, df_feat, target, is_train, feat_check=False, ignore_list=[], is_viz=True):


    if is_train==2:
        is_train_list = [1, 0]
    else:
        is_train_list = [is_train]

    for is_train in is_train_list:

        if is_train==1:
            tmp = df_feat[~df_feat[target].isnull()]
        elif is_train==0:
            tmp = df_feat[df_feat[target].isnull()]
        dir_path = '../features'

        length = len(tmp)
        if feat_check:
            for col in tmp.columns:
                if col in ignore_list:
                    continue

                # Nullがあるかどうか
                null_len = tmp[col].dropna().shape[0]
                if length - null_len>0:
                    print(f"{col}  | null shape: {length - null_len}")

                # infがあるかどうか
                max_val = tmp[col].max()
                min_val = tmp[col].min()
                if max_val==np.inf or min_val==-np.inf:
                    print(f"{col} | max: {max_val} | min: {min_val}")
            sys.exit()


        for col in tmp.columns:
            if col in ignore_list:
                continue

            feature = tmp[col].values.astype('float32')
            if is_train==1:
                feat_path = f'{dir_path}/train_{prefix}{col}'
            else:
                feat_path = f'{dir_path}/test_{prefix}{col}'

            if os.path.exists(feat_path): continue
            elif os.path.exists( f'../features/{col}'): continue
            else:
                if is_viz:
                    print(f"{feature.shape} | {col}")

                to_pkl_gzip(path=feat_path, obj=feature)


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


def get_kfold(valid_list, fold_seed, key='', target='', fold_n=5):

    #========================================================================
    # 1. Outlierの分類が困難なグループでKfoldを作る

    kfold_trn_list = []
    kfold_val_list = []

    for base_valid in valid_list:

        folds = KFold(n_splits=fold_n, shuffle=True, random_state=fold_seed)
        kfold = list(folds.split(base_valid, base_valid[target].values))

        # card_id listにする
        trn_list = []
        val_list = []
        for trn, val in kfold:

            "id ver"
            trn_ids = base_valid.iloc[trn][key].values
            val_ids = base_valid.iloc[val][key].values
            # idからindexを取得
            trn_ids = base_valid[base_valid[key].isin(trn_ids)].index.tolist()
            val_ids = base_valid[base_valid[key].isin(val_ids)].index.tolist()

            trn_list.append(trn_ids)
            val_list.append(val_ids)

            "index ver"
            #  trn_list.append(trn)
            #  val_list.append(val)

#         kfold_lazy = [trn_list, val_list]
#         kfold_list.append([trn_list, val_list])
        kfold_trn_list.append(trn_list)
        kfold_val_list.append(val_list)
    #========================================================================

    #========================================================================
    # kfoldの統合

    trn_list = []
    val_list = []
    # trainのfoldを作る
    for trn_fold in zip(*kfold_trn_list):
        tmp_list = []
        for tmp_fold in trn_fold:
            tmp_list += list(tmp_fold)
        trn_list.append(tmp_list)

    for val_fold in zip(*kfold_val_list):
        tmp_list = []
        for tmp_fold in val_fold:
            tmp_list += list(tmp_fold)
        val_list.append(tmp_list)

    kfold = [trn_list, val_list]
    #========================================================================

    return kfold


def max_freq_impute(df, col):
    max_freq = df[~df[col].isnull()][col].mode().values[0]
    return df[col].fillna(max_freq)


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]
