from google.cloud import storage as gcs
from pathlib import Path

HOME = Path().home()
gcp_credentials = HOME / 'privacy' / 'gixo-dim-horikoshi-all-gcp.json'


def gcs_upload(blob_name, file_path, bucket_name='horikoshi'):
    gcs_cl = gcs.Client.from_service_account_json(project='gixo-dim', json_credentials_path=gcp_credentials)
    
    bucket = gcs_cl.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(file_path))