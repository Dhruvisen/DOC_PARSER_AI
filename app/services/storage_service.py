import os
import uuid
import boto3
from botocore.client import Config
from pathlib import Path
from typing import Tuple, Optional
from app.core.config import get_settings

settings = get_settings()

class StorageService:
    def __init__(self):
        self.endpoint = settings.MINIO_ENDPOINT
        self.access_key = settings.MINIO_ACCESS_KEY
        self.secret_key = settings.MINIO_SECRET_KEY
        self.bucket_name = settings.MINIO_BUCKET_NAME
        self.secure = settings.MINIO_SECURE

        self.s3 = boto3.client(
            's3',
            endpoint_url=f"http://{self.endpoint}" if not self.secure else f"https://{self.endpoint}",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1' # MinIO ignores this but boto3 needs it
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except:
            print(f"Bucket {self.bucket_name} not found. Creating it...")
            self.s3.create_bucket(Bucket=self.bucket_name)

    def save_file(self, file_bytes: bytes, filename: str, user_id: str) -> str:
        """
        Saves a file to MinIO and returns a unique doc_id.
        Path: {user_id}/{doc_id}.{ext}
        """
        doc_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix.lower().replace(".", "")
        if not file_extension:
            file_extension = "bin"
            
        object_name = f"{user_id}/{doc_id}.{file_extension}"
        
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=object_name,
            Body=file_bytes,
            Metadata={'original_filename': filename}
        )
        
        return doc_id

    def get_file(self, doc_id: str, user_id: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Retrieves file bytes and type from MinIO.
        """
        prefix = f"{user_id}/{doc_id}"
        
        try:
            # Find the actual key (since we don't know the extension for sure here)
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' not in response:
                return None, None
                
            actual_key = response['Contents'][0]['Key']
            file_type = Path(actual_key).suffix.replace(".", "")
            
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=actual_key)
            return obj['Body'].read(), file_type
            
        except Exception as e:
            print(f"Error retrieving file from MinIO: {e}")
            return None, None

storage_service = StorageService()
