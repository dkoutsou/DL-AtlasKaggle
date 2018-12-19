import tensorflow as tf

from data_loader.data_generator import DataTestLoader
from models.DeepYeast_model import DeepYeastModel
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
        exit(0)

    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    testIterator = DataTestLoader(config)
    # create an instance of the model you want
    model = DeepYeastModel(config)
    # load model if exists
    model.load(sess)
    # here you predict from your model
    predictor = Predictor(sess, model, config)
    predictor.predict(testIterator)


if __name__ == '__main__':
    main()
