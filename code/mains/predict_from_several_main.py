import tensorflow as tf

from data_loader.data_generator import DataTestLoader
from models.models import all_models
from utils.config import process_config
from utils.utils import get_args
from utils.predictor import Predictor, get_pred_from_probas_threshold
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


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
        config_array = [process_config(x) for x in args.config.split(" ")]
        check_array = args.checkpoint_nb.split(" ")
        cwd = os.getenv("EXP_PATH")
        if args.outfile_multiple:
            outfile = cwd + '/' + args.outfile_multiple + '.csv'
        else:
            outfile = cwd + '/prediction.csv'
    except Exception:
        print("missing or invalid arguments")
        raise

    # not needed just to question n
    testIterator = DataTestLoader(config_array[0])
    probas = np.zeros((len(config_array), testIterator.n, 28))
    i = 0
    for config, check in zip(config_array, check_array):
        # create tensorflow session
        sess = tf.Session()
        # create your data generator
        # here config file used for init does not matter
        testIterator = DataTestLoader(config)
        # create an instance of the model you want
        try:
            ModelInit = all_models[config.model]
            model = ModelInit(config)
        except AttributeError:
            print("The model to use is not specified in the config file")
            exit(1)

        # load model if exists
        model.load(sess, check)
        # here you predict from your model
        predictor = Predictor(sess, model, config)
        probas[i, :, :] = predictor.predict_probas(testIterator)
        print('processed {} model'.format(model))
        i += 1
        tf.reset_default_graph()
    probas = np.mean(probas, axis=0)
    print(np.shape(probas))
    one_hot_pred = get_pred_from_probas_threshold(probas)
    bin = MultiLabelBinarizer(classes=np.arange(28))
    bin.fit([[1]])  # needed for instantiation of the object
    pred = bin.inverse_transform(one_hot_pred)
    predicted_labels = [
        ' '.join([str(p) for p in sample_pred]) for sample_pred in pred
    ]
    print(np.shape(predicted_labels))
    testIterator.result['Predicted'] = predicted_labels
    testIterator.result = testIterator.result.sort_values(by='Id')
    testIterator.result.to_csv(outfile, index=False)


if __name__ == '__main__':
    # if you want to specify the model number manually
    # because on other machine than training machine
    # get latest checkpoint does not work
    main()
