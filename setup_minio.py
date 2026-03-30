import boto3
from botocore.client import Config

def create_bucket():
    s3 = boto3.resource('s3',
                    endpoint_url='http://localhost:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')

    buckets = ['dvc-bucket', 'mlflow']
    for bucket_name in buckets:
        try:
            s3.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")
        except Exception as e:
            print(f"Error creating '{bucket_name}': {e}")

if __name__ == "__main__":
    create_bucket()
