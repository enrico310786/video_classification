import boto3
import os

#AWS_BUCKET = 'video_classification'
#OUTPUT_DIR = "downloads"
#DIRECTORY = 'ucf_action_sport/'


def multiup(aws_bucket, aws_directory, path_dir):
    print("Start the upload")
    upload_dir(aws_bucket, aws_directory, path_dir)

def upload_dir(aws_bucket, aws_directory, path_dir):
    client = boto3.client('s3')

    for root, directories, files in os.walk(path_dir):
        for name in files:

            local_path = os.path.join(root, name)
            s3_path = os.path.join(aws_directory, "/".join(local_path.split("/")[1:]))

            print("Upload '{}' to '{}'".format(local_path, s3_path))
            client.upload_file(local_path, aws_bucket, s3_path)