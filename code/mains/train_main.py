import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.DeepYeast_model import DeepYeastModel
from models.CP2_model import CP2Model
from models.CP4_model import CP4Model
from models.CDP4_model import CDP4Model
from models.CBDP4_model import CBDP4Model
from models.CDP2_model import CDP2Model
from models.CDP2D_model import CDP2DModel
from models.CBDP2_model import CBDP2Model
from models.SimpleCNN_model import SimpleCNNModel
#from models.inception_model import InceptionModel
from models.resNet_model import ResNetModel
from trainers.Network_trainer import NetworkTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


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
    # create tensorflow session
    configSess = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
    configSess.gpu_options.allow_growth = True
    sess = tf.Session(config=configSess)
    # create your data generator
    data = DataGenerator(config)
    # create an instance of the model you want
    try:
        if config.model == "DeepYeast":
            model = DeepYeastModel(config)
        elif config.model == "SimpleCNN":
            model = SimpleCNNModel(config)
        elif config.model == "CP2":
            model = CP2Model(config)
        elif config.model == "CP4":
            model = CP4Model(config)
        elif config.model == "CDP4":
            model = CDP4Model(config)
        elif config.model == "CBDP4":
            model = CBDP4Model(config)
        elif config.model == "CDP2":
            model = CDP2Model(config)
        elif config.model == "CDP2D":
            model = CDP2DModel(config)
        elif config.model == "CBDP2":
            model = CBDP2Model(config)
        elif config.model == "Inception":
            model = InceptionModel(config)
        elif config.model == "ResNet":
            model = ResNetModel(config)
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
    print('the gpu is avaiable {}'.format(
        tf.test.is_gpu_available()), flush=True)
    main()
