import boto3
import os
from settings import BUCKET_NAME


def download_s3_folder(bucket_name, s3_folder, local_dir='artifacts'):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    s3_prefix = s3_folder.lstrip('/')
    print(f"Searching in bucket: {bucket_name} with prefix: '{s3_prefix}'")

    found_anything = False
    for obj in bucket.objects.filter():
        print(obj.key)
        # Determine local path
        local_file_path = os.path.join(local_dir, obj.key).replace("sentence_transformer.model", "sentence_transformer")
        # Wiem, że to paskudne i tak się nie robi, ale to na szybko, żeby jakkolwiek zadziałało
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)


        print(f"Downloading {obj.key} → {local_file_path}")
        bucket.download_file(obj.key, local_file_path)

    if not found_anything:
        print("Error: No objects found matching that prefix. Check your S3 path.")
    print("Download complete.")
download_s3_folder(BUCKET_NAME, "/artifacts/model/sentence_transformer.model")