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
log_table_name        = CONFIG['log_table_name']
pred_value_table_name = start_time[4:13] + '__' + CONFIG['pred_value_table_name']
bucket_name           = CONFIG['bucket_name']
bucket_pred_dir_name  = CONFIG['bucket_pred_dir_name']
bucket_feature_dir_name = CONFIG['bucket_feature_dir_name']


def get_bq_client():
    gcp_config_path = f'{HOME}/privacy/gcp.yaml'
    with open(gcp_config_path, 'r') as f:
        gcp_config = yaml.load(f)
        
    is_create = False
    dataset_name = 'dim_ml_dataset'
    credentials  =  HOME + '/privacy/' + gcp_config['gcp_credentials']
    
    # Delete
    bq_client = BigQuery(credentials, dataset_name)
    
    return bq_client

#========================================================================
# CREATE TABLE
#========================================================================

def create_train_log_table(bq_client):

    schema_map = {
        'exp_date'        : 'STRING',
        'n_features'      : 'NUMERIC',
        'n_rows'          : 'NUMERIC',
        'cv_score'        : 'NUMERIC',
        'metric'          : 'STRING',
        'model_type'      : 'STRING',
        'learning_rate'   : 'FLOAT',
        'objective'       : 'NUMERIC',
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
    
    column_modes = ['REQUIRED'] * 7 + ['NULLABLE'] * (len(schema_map) - 7)
    
    schema = bq_client.create_schema(list(schema_map.keys()), list(schema_map.values()), column_modes)
    bq_client.create_table(log_table_name, schema)
    
    
    
def create_pred_value_table(bq_client, COLUMNS_EXP):

    schema_map = {}
    for col in COLUMNS_EXP:
        schema_map[col] = 'FLOAT'
    
    schema = bq_client.create_schema(list(schema_map.keys()), list(schema_map.values()))
    bq_client.create_table(pred_value_table_name, schema)
    
    
def insert_to_bq(bq_client, table_name, bucket_dir_name, df_log):
    
    storage_client = gcs.Client(project_name)

    fname = 'tmp.csv'
    file_path = '../log/' + fname
    df_log.to_csv(file_path, index=False)
    
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(fname)
    blob.upload_from_filename(filename=file_path)
    
    if bucket_dir_name:
        blob_name = bucket_dir_name + '/' + fname
    else:
        blob_name = fname
    bq_client.insert_from_gcs(table_name, bucket_name, blob_name)
    
    
def log_insert(bq_client, df_log):
    insert_to_bq(bq_client, log_table_name, '', df_log)
    
    
def save_train_log(log_map):
    """Summary line.
    log_map['exp_date']   = start_time
    log_map['n_features'] = train_df.shape[1]
    log_map['n_rows']     = train_df.shape[0]
    log_map['cv_score']   = cv_score
    log_map['metric']     = metric
    log_map['model_type'] = model_type
    
    Args:
    
    Returns:
    """
    bq_client = get_bq_client()
    table_ref = bq_client.dataset.table(log_table_name)
    table = bq_client.client.get_table(table_ref)
    columns_table = [field.name for field in table.schema]
    
    remain_cols = list(set(columns_table) - set(log_map.keys()))
    
    for col in remain_cols:
        log_map[col] = params.get(col)
    df_log = pd.Series(log_map).to_frame().T
    log_insert(bq_client, df_log)
    
    
def pred_table(bq_client, df_pred):
    create_pred_value_table(bq_client, df_pred.columns)
    insert_to_bq(pred_value_table_name, bucket_pred_dir_name, df_pred)