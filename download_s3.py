import boto3
import os


#AWS_BUCKET = 'video_classification'
#OUTPUT_DIR = "downloads"
#DIRECTORY = 'ucf_action_sport/'


def download_dir_new(aws_bucket, aws_directory):
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(aws_bucket)

    for obj in my_bucket.objects.filter(Prefix=aws_directory):

        output_file = obj.key.split('/')[-1]
        directory = obj.key.split('/')[1:-1]

        if output_file == "":
            continue
        else:
            directory = "/".join(directory)
            path_file = os.path.join(directory, output_file)

            #print('Download the file: ', obj.key)
            s3.Object(bucket_name=obj.bucket_name, key=obj.key).download_file(output_file)

            if not os.path.exists(directory):
                #print('Create the directory: ', directory)
                os.makedirs(directory)

            #print("Move the file '{}' to '{}'".format(output_file, path_file))
            os.replace(output_file, path_file)


def multidown(aws_bucket, aws_directory):
    print("Start the download")
    download_dir_new(aws_bucket, aws_directory)