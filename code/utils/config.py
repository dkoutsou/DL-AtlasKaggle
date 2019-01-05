import json
from bunch import Bunch
import os
import sys


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    experiment_dir = os.getenv("EXP_PATH")
    if experiment_dir is None:
        print("Set your EXP_PATH env first")
        sys.exit(1)
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join(experiment_dir, config.exp_name,
                                      "summary/")
    config.checkpoint_dir = os.path.join(experiment_dir, config.exp_name,
                                         "checkpoint/")
    return config


def generate_bagging_configs(config, num_of_configs):
    configs = []
    for i in range(num_of_configs):
        new_conf = Bunch(config.copy())
        new_conf.summary_dir = os.path.join(new_conf.summary_dir,
                                            'model{}/'.format(i))
        new_conf.checkpoint_dir = os.path.join(new_conf.checkpoint_dir,
                                               'model{}/'.format(i))
        configs.append(new_conf)
    return configs
