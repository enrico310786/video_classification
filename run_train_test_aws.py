import argparse
from download_s3 import multidown
import yaml

from train_test_classification_model import run_train_test_model


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    parser.add_argument('--aws_directory', type=str, help='Aws bucket directory where I recreate the project structure. Put / at the end')
    parser.add_argument('--aws_bucket', type=str, help='aws bucket')
    opt = parser.parse_args()

    # 1 - Download directory from AWS bucket
    aws_bucket = opt.aws_bucket
    print('aws_bucket: ', aws_bucket)
    aws_directory = opt.aws_directory
    print('aws_directory: ', aws_directory)
    print('Download directories from S3')
    multidown(aws_bucket, aws_directory)

    # 2 - load config file
    path_config_file = opt.path_config_file
    print('path_config_file: ', path_config_file)
    cfg = load_config(path_config_file)

    # 3 - run train and test
    do_train = cfg.get("do_train", 1.0) > 0.0
    do_test = cfg.get("do_test", 1.0) > 0.0

    run_train_test_model(cfg=cfg,
                         do_train=do_train,
                         do_test=do_test,
                         aws_bucket=aws_bucket,
                         aws_directory=aws_directory)


