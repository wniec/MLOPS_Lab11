import boto3
import os
from .settings import BUCKET_NAME, MODEL_PATH
def download_s3_folder(bucket_name, s3_folder, local_dir='artifacts'):

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        # Skip if it's a folder
        if obj.key.endswith('/'):
            continue

        # Determine local path
        local_file_path = os.path.join(local_dir, obj.key.split('/')[-1])

        print(f"Downloading {obj.key} → {local_file_path}")
        bucket.download_file(obj.key, local_file_path)

    print("Download complete.")

download_s3_folder(BUCKET_NAME, MODEL_PATH)