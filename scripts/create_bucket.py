import boto3
from botocore.client import Config
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.resource('s3',
                    endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000'),
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minio'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minio123'),
                    config=Config(signature_version='s3v4'),
                    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))

bucket_name = 'mlflow'

try:
    s3.create_bucket(Bucket=bucket_name)
    print(f"Bucket '{bucket_name}' created successfully.")
except Exception as e:
    print(f"Error creating bucket: {e}")
