import tensorflow as tf

from data_loader.data_generator import DataTestLoader
from models.DeepYeast_model import DeepYeastModel
from models.CP2_model import CP2Model
from models.CP4_model import CP4Model
from models.CDP4_model import CDP4Model
from models.CBDP4_model import CBDP4Model
from models.CDP2_model import CDP2Model
from models.CBDP2_model import CBDP2Model
from models.SimpleCNN_model import SimpleCNNModel
from models.inception_model import InceptionModel
from models.resNet_model import ResNetModel
from models.kaggle_model import KaggleModel
from models.DeepSimple_model import DeepSimpleModel
from utils.config import process_config
from utils.utils import get_args
from utils.predictor import Predictor


def main():
    """ Loads the model from the checkpoint dir
    as specified in the given config file.
    Calls the prediction function to save the
    prediction csv file to the checkpoint dir.
    """
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception:
        print("missing or invalid arguments")
        raise


    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    testIterator = DataTestLoader(config)
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
        elif config.model == "CBDP2":
            model = CBDP2Model(config)
        elif config.model == "Inception":
            model = InceptionModel(config)
        elif config.model == "ResNet":
            model = ResNetModel(config)
        elif config.model == "Kaggle":
            model = KaggleModel(config)
        elif config.model == "DeepSimple":
            model = DeepSimpleModel(config)
    except AttributeError:
        print("The model to use is not specified in the config file")
        exit(1)
    
    # load model if exists
    model.load(sess, args.checkpoint_nb)
    # here you predict from your model
    predictor = Predictor(sess, model, config)
    predictor.predict(testIterator)


if __name__ == '__main__':
    # if you want to specify the model number manually
    # because on other machine than training machine
    # get latest checkpoint does not work
    main()
