import os
import sys
import yaml
import datetime
import numpy as np
import pandas as pd
from func.BigQuery import BigQuery
from google.cloud import storage as gcs


HOME = os.path.expanduser('~')
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())[:13]

ieee_config_path = f"{HOME}/github/ieee-fraud/config/ieee_config.yaml"
with open(ieee_config_path, 'r') as f:
    CONFIG = yaml.load(f)
    
project_name          = CONFIG['project_name']
dataset_name          = CONFIG['dataset_name']
log_table_name        = CONFIG['log_table_name']
pred_value_table_name = start_time[4:13] + '__' + CONFIG['pred_value_table_name']
bucket_name           = CONFIG['bucket_name']
bucket_pred_name      = CONFIG['bucket_pred_name']
bucket_feature_dir_name = CONFIG['bucket_feature_name']

"""
User Reference:

create_train_log_table()

log_map = {}
log_map['datetime']    = start_time
log_map['n_features']  = train_df.shape[1]
log_map['n_rows']      = train_df.shape[0]
log_map['cv_score']    = cv_score
log_map['fold1_score'] = score_list[0]
log_map['fold2_score'] = score_list[1]
log_map['fold3_score'] = score_list[2]
log_map['fold4_score'] = score_list[3]
log_map['fold5_score'] = score_list[4]
log_map['seed']        = seed
log_map['metric']      = metric
log_map['model_type']  = model_type
save_train_log(log_map)
"""

#========================================================================
# CREATE TABLE
#========================================================================

def create_train_log_table(bq):

    schema_map = {
        'datetime'        : 'STRING',
        'n_features'      : 'NUMERIC',
        'n_rows'          : 'NUMERIC',
        'cv_score'        : 'FLOAT',
        'adv_cv_score'    : 'FLOAT',
        'fold1_score'     : 'FLOAT',
        'fold2_score'     : 'FLOAT',
        'fold3_score'     : 'FLOAT',
        'fold4_score'     : 'FLOAT',
        'fold5_score'     : 'FLOAT',
        'best_iteration'  : 'NUMERIC',
        'seed'            : 'NUMERIC',
        'metric'          : 'STRING',
        'model_type'      : 'STRING',
        'learning_rate'   : 'FLOAT',
        'objective'       : 'STRING',
        'num_leaves'      : 'NUMERIC',
        'max_depth'       : 'NUMERIC',
        'subsample'       : 'FLOAT',
        'subsample_freq'  : 'FLOAT',
        'colsample_bytree': 'FLOAT',
        'lambda_l1'       : 'FLOAT',
        'lambda_l2'       : 'FLOAT',
        'batch_size'      : 'NUMERIC',
        'n_epochs'        : 'NUMERIC',
    }
    mode_map = {
        'datetime'        : 'REQUIRED',
        'n_features'      : 'REQUIRED',
        'n_rows'          : 'REQUIRED',
        'cv_score'        : 'REQUIRED',
        'adv_cv_score'    : 'NULLABLE',
        'fold1_score'     : 'REQUIRED',
        'fold2_score'     : 'REQUIRED',
        'fold3_score'     : 'REQUIRED',
        'fold4_score'     : 'REQUIRED',
        'fold5_score'     : 'REQUIRED',
        'best_iteration'  : 'NULLABLE',
        'seed'            : 'REQUIRED',
        'metric'          : 'REQUIRED',
        'model_type'      : 'REQUIRED',
        'learning_rate'   : 'REQUIRED',
        'objective'       : 'REQUIRED',
        'num_leaves'      : 'NULLABLE',
        'max_depth'       : 'NULLABLE',
        'subsample'       : 'NULLABLE',
        'subsample_freq'  : 'NULLABLE',
        'colsample_bytree': 'NULLABLE',
        'lambda_l1'       : 'NULLABLE',
        'lambda_l2'       : 'NULLABLE',
        'batch_size'      : 'NULLABLE',
        'n_epochs'        : 'NULLABLE',
    }
    
    schema = bq.create_schema(list(schema_map.keys()), list(schema_map.values()), list(mode_map.values()))
    bq.create_table(log_table_name, schema)
    
    
def create_new_log_value(bq, new_column_name, new_column_type):
    bq.create_new_field(log_table_name, new_column_name, new_column_type)
    
    
def create_pred_value_table(bq, COLUMNS_EXP):

    schema_map = {}
    mode_map = {}
    for col in COLUMNS_EXP:
        schema_map[col] = 'FLOAT'
        mode_map[col]   = 'REQUIRED'
    
    schema = bq.create_schema(list(schema_map.keys()), list(schema_map.values()), list(mode_map.values()))
    bq.create_table(pred_value_table_name, schema)
    
    
def insert_to_bq(bq, table_name, bucket_name, df_log, fname='tmp.csv'):
    
    storage_client = gcs.Client(project_name)
    file_path = '../log/' + fname
    df_log.to_csv(file_path, index=False)
    
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(fname)
    blob.upload_from_filename(filename=file_path)
    
    blob_name = fname
    bq.insert_from_gcs(table_name, bucket_name, blob_name)
    
    
def log_insert(bq, df_log):
    insert_to_bq(bq, log_table_name, bucket_name, df_log)
    
    
def save_train_log(log_map, model_params):
    """Summary line.
    trainingにおけるscoreやmodel parameterのログをBQへ保存する.
    各実験の予測値、特徴セットはexp_dateと予測値ファイル名、特徴ファイル名が紐づいてるので、  
    そこからGCS経由で取得できる.
    
    log_map['datetime']    = start_time
    log_map['n_features']  = train_df.shape[1]
    log_map['n_rows']      = train_df.shape[0]
    log_map['cv_score']    = cv_score
    log_map['fold1_score'] = score_list[0]
    log_map['fold2_score'] = score_list[1]
    log_map['fold3_score'] = score_list[2]
    log_map['fold4_score'] = score_list[3]
    log_map['fold5_score'] = score_list[4]
    log_map['seed']        = seed
    log_map['metric']      = metric
    log_map['model_type']  = model_type
    
    Args:
    
    Returns:
    """
    bq = BigQuery(dataset_name)
    table = bq.get_table(log_table_name)
    columns_table = [field.name for field in table.schema]
    
    remain_cols = list(set(columns_table) - set(log_map.keys()))
    
    for col in remain_cols:
        log_map[col] = model_params.get(col)

    df_log = pd.Series(log_map).to_frame().T
    df_log = df_log[columns_table]
    log_insert(bq, df_log)
    
    
def pred_table(df_pred):
    """Summary line.
    Lookerで分析を行いたい予測値データセットでBQテーブルを作成する
    
    Args:
    
    Returns:
    """
    create_pred_value_table(df_pred.columns)
    insert_to_bq(pred_value_table_name, bucket_name, df_pred)
    
    
def del_pred_tables(del_startswith):
    """Summary line.
    予測値データセットを前方一致でまとめて削除する
    
    Args:
    
    Returns:
    """
    bq_client = BigQuery.client
    all_tables = bq_client.get_list_table()
    multi_del_tables(all_tables, del_startswith)