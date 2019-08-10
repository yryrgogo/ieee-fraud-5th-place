import os
import yaml
from time import sleep
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger


HOME = os.path.expanduser('~')

class BigQuery:

    def __init__(self, dataset_name='', gcp_config_path='', is_create=False, OUTPUT_DIR='../output'):

        # Config
        #  if len(gcp_config_path)==0:
        #      gcp_config_path = f'{HOME}/privacy/gcp.yaml'
        #  with open(gcp_config_path, 'r') as f:
        #      gcp_config = yaml.load(f)

        #  if os.path.isdir('../config'):
        #      credentials  =  '../config/' + gcp_config['gcp_credentials']
        #  else:
        #      credentials   = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
#             credentials  =  f'{HOME}/privacy/' + gcp_config['gcp_credentials']
        credentials   = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

        # self.client = bigquery.Client()
        self.client = bigquery.Client.from_service_account_json(credentials)
        self.table_dict = {}

        if dataset_name:
            self.dataset_name = dataset_name
            if not is_create:
                self.set_dataset(dataset_name)

    def set_dataset(self, dataset_name):
        self.dataset_name = dataset_name
        dataset_ref = self.client.dataset(self.dataset_name)
        self.dataset = self.client.get_dataset(dataset_ref)
        print('Setup Dataset {}.'.format(self.dataset.dataset_id))

    def set_table(self, table_name):
        table_ref = self.dataset.table(table_name)
        self.table_dict[table_name] = self.client.get_table(table_ref)
        print('Setup Table {}.'.format(self.table_dict[table_name].table_id))

    def create_dataset(self):
        dataset_ref = self.client.dataset(self.dataset_name)
        dataset = bigquery.Dataset(dataset_ref)
        self.dataset = self.client.create_dataset(dataset)

        print('Dataset {} created.'.format(self.dataset.dataset_id))

    def create_table(self, table_name, schema):

        table_ref = self.dataset.table(table_name)
        table = bigquery.Table(table_ref, schema=schema)
        self.table_dict[table_name] = self.client.create_table(table)

        print('Table {} created.'.format(self.table_dict[table_name].table_id))

    def create_schema(self, column_names, column_types, column_modes):
        schema = []
        for col_name, col_type, col_mode in zip(column_names, column_types, column_modes):
            schema.append(bigquery.SchemaField(col_name, col_type, mode=col_mode))
        return schema

    def insert_rows(self, table_name, insert_rows):
        res = self.client.insert_rows(self.table_dict[table_name], insert_rows)
        if res:
            print("Insert Error!!: {}".format(res))

    def del_table(self, table_name):

        dataset_ref = self.client.dataset(self.dataset_name)
        table_ref = self.dataset.table(table_name)
        res = self.client.delete_table(table_ref)
        print("del table: {} | Res: {}".format(table_ref, res))

    def del_dataset_all(self):

        dataset_ref = self.client.dataset(self.dataset_name)
        table_ref_list = list(self.client.list_tables(dataset_ref))

        for table_ref in table_ref_list:
            self.client.delete_table(table_ref)
            print("del table: {}".format(table_ref))
        self.client.delete_dataset(dataset_ref)
        print("del dataset: {}".format(dataset_ref))

    def insert_from_gcs(self, table_name, bucket_name, blob_name, source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1):

        table_ref = self.dataset.table(table_name)

        self.gcs_url = "gs://{}/{}".format(bucket_name, blob_name)

        job_id_prefix = 'go_job'
        job_config = bigquery.LoadJobConfig()
        job_config.skip_leading_rows = skip_leading_rows
        job_config.source_format = source_format

        load_job = self.client.load_table_from_uri(
            self.gcs_url,
            table_ref,
            job_config=job_config,
            job_id_prefix=job_id_prefix
        )

#         assert load_job.state == 'RUNNING'
#         assert load_job.job_type == 'load'

        load_job.result()  # Waits for table load to complete

#         assert load_job.state == 'DONE'
        assert load_job.job_id.startswith(job_id_prefix)


    def get_query_result(self, query):
        query_job = self.client.query(query)
        row_list = []
        for row in query_job:
            row_list.append([*list(row)])

        if len(row_list) == 0:
            return pd.DataFrame()

        result = pd.DataFrame(row_list, columns=list(row.keys()))
        result.reset_index(drop=True, inplace=True)
        return result


    def del_rows_query(self, query):
        """
        Sample Query:
        query = (
            f'''
            DELETE FROM {dataset_name}.{table_name}
            WHERE {column_name} in ('{delete_name}')
            ''')
       """

        query_job = self.client.query(query)
        print("Done! {query_job}")


    def create_new_field(self, table_name, new_column_name, new_column_type):
        table_ref = self.dataset.table(table_name)
        table = self.client.get_table(table_ref)
        original_schema = table.schema
        new_schema = original_schema[:]  # creates a copy of the schema
        new_schema.append(bigquery.SchemaField(new_column_name, new_column_type, "NULLABLE"))

        table.schema = new_schema
        table = self.client.update_table(table, ["schema"])  # API request

        assert len(table.schema) == len(original_schema) + 1 == len(new_schema)

    
    def get_list_table(self):
        all_tables = self.client.list_tables(self.dataset)
        all_tables = [table.table_id for table in all_tables]
        return all_tables
    
    
    def multi_del_tables(self, del_table_names, del_startswith):
        for del_table_name in del_table_names:
            if del_table_name.startswith(del_startswith):
                self.client.del_table(del_table_name)

                
    def create_table_from_query(self, dataset_name, table_name, query):
        job_config = bigquery.QueryJobConfig()
        table_ref = self.client.dataset(dataset_name).table(table_name)
        job_config.destination = table_ref
        
        query_job = self.client.query(
            query,
            location='US',
            job_config=job_config,
        )
        query_job.result()
        print("* query result to: ", table_ref.path)