import argparse
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
    opt = parser.parse_args()

    # 1 - load config file
    path_config_file = opt.path_config_file
    print('path_config_file: ', path_config_file)
    cfg = load_config(path_config_file)

    # 2 - run train and test
    do_train = cfg.get("do_train", 1.0) > 0.0
    do_test = cfg.get("do_test", 1.0) > 0.0
    run_train_test_model(cfg=cfg,
                         do_train=do_train,
                         do_test=do_test)
