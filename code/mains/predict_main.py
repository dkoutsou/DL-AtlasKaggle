import tensorflow as tf

from data_loader.data_generator import DataTestLoader
from models.models import all_models
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
        ModelInit = all_models[config.model]
        model = ModelInit(config)
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
