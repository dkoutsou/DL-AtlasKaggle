import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.models import all_models
from trainers.Network_trainer import NetworkTrainer
from utils.config import process_config, generate_bagging_configs
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

# WORK IN PROGRESS...


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception:
        print("missing or invalid arguments")
        raise

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create model configs and dirs
    model_configs = generate_bagging_configs(config, config.n_estimators)
    for c in model_configs:
        create_dirs([c.summary_dir, c.checkpoint_dir])

    # create tensorflow session
    configSess = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    configSess.gpu_options.allow_growth = True
    sess = tf.Session(config=configSess)

    # create your data generator
    data = DataGenerator(config)

    # create an instance of the model you want
    try:
        ModelConstructor = all_models[config.model]
        model = ModelConstructor(config)
    except AttributeError:
        print("The model to use is not specified in the config file")
        exit(1)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = NetworkTrainer(sess, model, data, config, logger)

    # load model if exists
    model.load(sess)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    print(
        'the gpu is avaiable {}'.format(tf.test.is_gpu_available()),
        flush=True)
    main()
