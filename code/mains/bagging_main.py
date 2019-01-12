import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.models import all_models
from models.bagging_model import BaggingModel
from trainers.bagging_trainer import BaggingTrainer
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
    master_config = config

    # create tensorflow sessions
    sessions = []
    for i in range(config.n_estimators):
        configSess = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        configSess.gpu_options.allow_growth = True
        sess = tf.Session(config=configSess)
        sessions.append(sess)
    configSess = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    configSess.gpu_options.allow_growth = True
    master_sess = tf.Session(config=configSess)

    # create your data generator
    data_gens = []
    for i in range(config.n_estimators):
        # Make sure they all use different random seed
        data = DataGenerator(config, random_state=42 + i)
        data_gens.append(data)

    # create an instances of the model you want
    models = []
    try:
        ModelConstructor = all_models[config.model]
        for i in range(config.n_estimators):
            model = ModelConstructor(config)
            models.append(model)
    except AttributeError:
        print("The model to use is not specified in the config file")
        exit(1)
    master_model = BaggingModel(config, models)

    # create tensorboard loggers for each model and session
    loggers = []
    for sess, config in zip(sessions, model_configs):
        logger = Logger(sess, config)
        loggers.append(logger)
    master_logger = Logger(master_sess, master_config)

    # create trainer and pass all the previous components to it
    trainer = BaggingTrainer(sessions, models, data_gens, model_configs,
                             loggers, master_sess, master_model, master_config,
                             master_logger)

    # load models if exists
    for m, s in zip(models, sessions):
        m.load(s)
    master_model.load(master_sess)

    # here you train your models
    trainer.train()


if __name__ == '__main__':
    print(
        'the gpu is avaiable {}'.format(tf.test.is_gpu_available()),
        flush=True)
    main()
