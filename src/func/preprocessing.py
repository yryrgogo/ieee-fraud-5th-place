import pandas as pd
import numpy as np
import sys, re, glob
import gc
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool
import multiprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import category_encoders as ce


sc = StandardScaler()
mm = MinMaxScaler()

""" 前処理系 """

' データセットを標準化、欠損値、無限値の中央値置き換え '
def data_regulize(df, na_flg=0, inf_flg=0, sc_flg=0, mm_flg=0, float16_flg=0, ignore_feature_list=[], logger=False):

    if inf_flg==1:
        df = inf_replace(df=df, logger=logger, ignore_feature_list=ignore_feature_list)
    if na_flg==1:
        df = impute_avg(df=df, logger=logger, ignore_feature_list=ignore_feature_list)

    ' 正規化 / 標準化 '
    if sc_flg==1:
        #  df[col] = sc.fit_transform(df[col])
        avg = df[col].mean()
        se = df[col].std()
        df[col] = (df[col] - avg) / se
    elif mm_flg==1:
        df = max_min_regularize(df, ignore_feature_list=ignore_feature_list, logger=logger)
    if float16_flg==1:
        df = df.astype('float16')

    return df


def impute_avg(df, logger=False, drop=False, ignore_feature_list=[]):
    for col in df.columns:
        if col in ignore_feature_list:continue
        if len(df[col][df[col].isnull()])>0:
            if drop:
                df.drop(col, axis=1, inplace=True)
                if logger:
                    logger.info(f'drop: {col}')
                continue
            df[col] = df[col].fillna(df[col].mean())
            if logger:
                logger.info(f'{col} impute length: {len(df[col][df[col].isnull()])}')

    return df


def inf_replace(df, logger=False, drop=False, ignore_feature_list=[]):
    for col in df.columns:
        if col in ignore_feature_list:continue

        ' count of inf '
        inf_plus = np.where(df[col].values == float('inf') )
        inf_minus = np.where(df[col].values == float('-inf') )
        logger.info(f'{col} >> inf count: {len(inf_plus)} | -inf count: {len(inf_minus)}')

        df[col].replace(np.inf, np.nan, inplace=True)
        df[col].replace(-1*np.inf, np.nan, inplace=True)
        logger.info(f'*****inf replace SUCCESS!!*****')
    return df

    #      for i in range(len(inf_plus[0])):
    #          logger.info(f'inf include: {col}')
    #          df[col].values[inf_plus[0][i]] = np.nan
    #          df[col].values[inf_minus[0][i]] = np.nan
    #          logger.info(f'-inf include: {col}')

    #  return df


def max_min_regularize(df, ignore_feature_list=[], logger=False):
    for col in df.columns:
        if col in ignore_feature_list:continue
        #  try:
        #      df[col] = mm.fit_transform(df[col].values)
        #  except TypeError:
        #      if logger:
        #          logger.info('TypeError')
        #          logger.info(df[col].drop_duplicates())
        #  except ValueError:
        #      if logger:
        #          logger.info('ValueError')
        #          logger.info(df[col].shape)
        #          logger.info(df[col].head())
        c_min = df[col].min()
        if c_min<0:
            df[col] = df[col] + np.abs(c_min)
        c_max = df[col].max()
        df[col] = df[col] / c_max

    return df


' 外れ値除去 '
def outlier(df, col=False, out_range=1.64, print_flg=False, replace_value=False, drop=False, replace_inner=False, logger=False, plus_replace=True, minus_replace=True, plus_limit=False, minus_limit=False, z_replace=False):
    '''
    Explain:
    Args:
        df(DF)        : 外れ値を除外したいデータフレーム
        col(float)    : 標準偏差を計算する値
        out_range(float): 外れ値とするZ値の範囲.初期値は1.96としている
    Return:
        df(DF): 入力データフレームから外れ値を外したもの
    '''

    std = df[col].std()
    avg = df[col].mean()

    tmp_val = df[col].values
    z_value = (tmp_val - avg)/std
    df['z'] = z_value

    inner = df[df['z'].between(left=-1*out_range, right=out_range)]
    plus_out  = df[df['z']>out_range]
    minus_out = df[df['z']<-1*out_range]

    if logger:
        length = len(df)
        in_len = len(inner)
        logger.info(f'''
#==========================================
# column        : {col}
# out_range     : {out_range}
# replace_value : {replace_value}
# plus_replace  : {plus_replace}
# minus_replace : {minus_replace}
# z_replace     : {z_replace}
# plus_limit    : {plus_limit}
# minus_limit   : {minus_limit}
# drop          : {drop}
# all max       : {df[col].max()}
# inner  max    : {inner[col].max()}
# all min       : {df[col].min()}
# inner  min    : {inner[col].min()}
# all length    : {length}
# inner length  : {in_len}
# diff length   : {length-in_len}
# plus out len  : {len(plus_out)}
# minus out len : {len(minus_out)}
#==========================================
        ''')

    # replace_valueを指定してz_valueを使い置換する場合
    if replace_value:
        if z_replace:
            if plus_replace:
                df[col] = df[col].where(df['z']<=out_range, replace_value)
            if minus_replace:
                df[col] = df[col].where(df['z']>=-out_range, replace_value)
        if plus_limit:
            df[col] = df[col].where(df['z']<=plus_limit, replace_value)
        if minus_limit:
            df[col] = df[col].where(df['z']>=-minus_limit, replace_value)


    # 外れ値を除去する場合
    elif drop:
        if plus_replace:
            df = df[df['z']>=-1*out_range]
        elif minus_replace:
            df = df[df['z']<=out_range]
    # replace_valueを指定せず、innerのmax, minを使い有意水準の外を置換する場合
    elif replace_inner:
        inner_max = inner[col].max()
        inner_min = inner[col].min()
        if plus_replace:
            df[col] = df[col].where(df['z']<=out_range, inner_max)
        elif minus_replace:
            df[col] = df[col].where(df['z']>=-out_range, inner_min)

    plus_out_val  = df[df['z']>out_range][col].drop_duplicates().values
    minus_out_val = df[df['z']<-1*out_range][col].drop_duplicates().values
    if logger:
        if df[col].max() > inner[col].max() and df[col].min() < inner[col].min():
            logger.info(f'''
#==========================================
# RESULT
# plus out value  : {np.max(plus_out_val)}
# minus out value : {np.min(minus_out_val)}
#==========================================
    ''')
        elif df[col].max() == inner[col].max() and df[col].min() < inner[col].min():
            logger.info(f'''
#==========================================
# RESULT
# plus out value  : Nothing
# minus out value : {np.min(minus_out_val)}
#==========================================
    ''')
        elif df[col].max() > inner[col].max() and df[col].min() == inner[col].min():
            logger.info(f'''
#==========================================
# RESULT
# plus out value  : {np.max(plus_out_val)}
# minus out value : Nothing
#==========================================
    ''')

    del plus_out, minus_out
    gc.collect()

    return df


def contraction(df, value, limit, max_flg=1, nan_flg=0):
    '''
    Explain:
        収縮法。limitより大きいor小さい値をlimitの値で置換する。
    Args:
        df    :
        value   :
        limit   : 収縮を適用する閾値
        max_flg : limitより大きい値を収縮する場合は1。小さい値を収縮する場合は0。
        null_flg: limitの値ではなくNaNに置換する場合は1。
    Return:
    '''

    if max_flg==1 and nan_flg==0:
        df[value] = df[value].map(lambda x: limit if x > limit else x)
    elif max_flg==0 and nan_flg==0:
        df[value] = df[value].map(lambda x: limit if x < limit else x)
    elif max_flg==1 and nan_flg==1:
        df[value] = df[value].map(lambda x: np.nan if x > limit else x)
    elif max_flg==0 and nan_flg==1:
        df[value] = df[value].map(lambda x: np.nan if x < limit else x)

    return df


#  def impute_avg(data=None, unique_id=none, level=None, index=1, value=None):
#      '''
#      Explain:
#          平均値で欠損値補完を行う
#      Args:
#          data(DF)       : nullを含み、欠損値補完を行うデータ
#          level(list)    : 集計を行う粒度。最終的に欠損値補完を行う粒度が1カラム
#                           でなく、複数カラム必要な時はリストで渡す。
#                           ただし、欠損値補完の集計を行う粒度がリストと異なる場合、
#                           次のindex変数にリストのうち集計に使うカラム数を入力する
#                           (順番注意)
#          index(int)     : 欠損値補完の際に集計する粒度カラム数
#          value(float)   : 欠損値補完する値のカラム名
#      Return:
#          result(DF): 欠損値補完が完了したデータ
#      '''

#      ' 元データとの紐付けをミスらない様にインデックスをセット '
#      #  data.set_index(unique_id, inplace=true)

#      ' Null埋めする為、level粒度の平均値を取得 '
#      use_cols = level + [value]
#      data = data[use_cols]
#      imp_avg = data.groupby(level, as_index=False)[value].mean()

#      ' 平均値でNull埋め '
#      null = data[data[value].isnull()]
#      #  null = null.reset_index()
#      fill_null = null.merge(imp_avg, on=level[:index], how='inner')

#      ' インデックスをカラムに戻して、Null埋めしたDFとconcat '
#      data = data[data[value].dropna()]
#      result = pd.concat([data, fill_null], axis=0)

#      return result


def lag_feature(df, col_name, lag, level=[]):
    '''
    Explain:
        対象カラムのラグをとる
        時系列データにおいてリーケージとなる特徴量を集計する際などに使用
    Args:
        df(DF)    : valueやlevelを含んだデータフレーム
        col_name    : ラグをとる特徴量のカラム名
        lag(int)    : shiftによってずらす行の数
                     （正：前の行数分データをとる、 負：後の行数分データをとる）
        level(list) : 粒度を指定してラグをとる場合、その粒度を入れたリスト。
                      このlevelでgroupbyをした上でラグをとる
    Return:
        df: 最初の入力データにラグをとった特徴カラムを付与して返す
    '''

    if len(level)==0:
        df[f'shift{lag}_{value}'] = df[col_name].shift(lag)
    else:
        df[f'shift{lag}_{value}@{level}'] = df.groupby(level)[col_name].shift(lag)

    return df


def ordinal_encode(df, to_cat_list):

    '''
    Explain:
    Args:
        to_cat_list: Eoncodeしたい列をリストで指定。複数指定可能。
    Return:
    '''

    # 序数をカテゴリに付与して変換
    ce_oe = ce.OrdinalEncoder(cols=to_cat_list, handle_unknown='impute')
    return ce_oe.fit_transform(df), ce_oe


def get_ordinal_mapping(obj):
    '''
    Explain:
        Ordinal Encodingの対応をpd.DataFrameで返す
    Args:
        obj : category_encodersのインスタンス
    Return:
        dframe
    '''
    tmp_list = list()
    for x in obj.category_mapping:
        tmp_list.extend([tuple([x['col']])+ i for i in x['mapping']])
    df_ord_map = pd.DataFrame(tmp_list, columns=['column','label','ord_num'])
    return df_ord_map


# カテゴリ変数をファクトライズ (整数に置換)する関数
def factorize_categoricals(df, cats, is_sort=True):
    for col in cats:
        df[col], _ = pd.factorize(df[col], sort=is_sort)
    return df


# カテゴリ変数のダミー変数 (二値変数化)を作成する関数
def get_dummies(df, cat_list, drop=True):

    before_cols = list(df.columns.values)

    for col in cat_list:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    if drop:
        df.drop(cat_list, axis=1, inplace=True)

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
    return df


def split_dataset(df, val_no, val_col='valid_no'):
    """
    時系列用のtrain, testデータを切る。validation_noを受け取り、
    データセットにおいてその番号をもつ行をTestデータ。その番号を
    もたない行をTrainデータとする。

    Args:
        df(DF): TrainとTestに分けたいデータセット
        val_no(int): 時系列においてleakが発生しない様にデータセットを
                     切り分ける為の番号。これをもとにデータを切り分ける

    Return:
        train(df): 学習用データフレーム(validationカラムはdrop)
        test(df) : 検証用データフレーム(validationカラムはdrop)
    """

    train = df[df[val_col] != val_no]
    test = df[df[val_col] == val_no]

    for col in train.columns:
        if col.count('valid_no'):
            train.drop(col, axis=1, inplace=True)
    for col in test.columns:
        if col.count('valid_no'):
            test.drop(col, axis=1, inplace=True)

    return train, test


def set_validation(df, target, unique_id, val_col='valid_no', fold=5, seed=1208, holdout_flg=0):
    '''
    Explain:
        データセットにvalidation番号を振る。繰り返し検証を行う際、
        validationを固定したいのでカラムにする。
    Args:
    Return:
    '''
    #  start_date = pd.to_datetime('2017-03-12')
    #  end_date = pd.to_datetime('2017-04-22')
    #  df['validation'] = df['visit_date'].map(lambda x: 1 if start_date <= x and x <= end_date else 0)

    if holdout_flg==1:
        ' 全体をStratifiedKFoldで8:2に切って、8をCV.2をHoldoutで保存する '

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        x = df.drop(target, axis=1)
        y = df[target].values

        for trn_idx, val_idx in cv.split(x, y):
            df.iloc[trn_idx].to_csv('../df/cv_app_train.csv', index=False)
            df.iloc[val_idx].to_csv('../df/holdout_app_train.csv', index=False)
            sys.exit()

    else:
        ' データをfold数に分割してvalidation番号をつける '
        cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
        df = df[[unique_id, target]].reset_index(drop=True)
        x = df[unique_id].to_frame()
        y = df[target].values
        cnt=0

        for trn_idx, val_idx in cv.split(x, y):
            cnt+=1

            valid_no = np.zeros(len(val_idx))+cnt
            tmp = pd.DataFrame({'index':val_idx, val_col:valid_no})

            if cnt==1:
                tmp_result = tmp.copy()
            else:
                tmp_result = pd.concat([tmp_result, tmp], axis=0)

        tmp_result.set_index('index', inplace=True)


        result = df.join(tmp_result)
        result.drop(target, axis=1, inplace=True)
        print(result.shape)
        print(result.head())

    return result


def squeeze_target(df, col_name, size):
    '''
    col_nameの各要素について、一定数以上のデータがある要素の行のみ残す
    '''

    tmp = df.groupby(col_name).size()
    target_id = tmp[tmp >= size].index

    df = df.set_index(col_name)
    result = df.loc[target_id, :]
    result = result.reset_index()
    return result

