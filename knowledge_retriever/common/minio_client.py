from minio import Minio
from minio.error import S3Error

class MinioClient:
    def __init__(self, endpoint, access_key, secret_key, secure=True):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def list_buckets(self):
        return self.client.list_buckets()

    def list_objects(self, bucket_name):
        return self.client.list_objects(bucket_name)

    def upload_file(self, bucket_name, object_name, file_path):
        try:
            result = self.client.fput_object(bucket_name, object_name, file_path)
            return result
        except S3Error as err:
            print(f"Error: {err}")

    def download_file(self, bucket_name, object_name, file_path):
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
        except S3Error as err:
            print(f"Error: {err}")