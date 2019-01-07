from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

def get_pred_from_probas(probas):
    tmp_pred = np.round(probas)
    for i in range(len(probas)):
        # if no classes predicted
        # i.e. no probas > 0.5
        # then choose the most probable one.
        if np.sum(tmp_pred[i]) == 0:
            try:
                tmp_pred[i, np.argmax(probas[i])[0]] = 1
            except IndexError:
                tmp_pred[i, np.argmax(probas[i])] = 1
        # more than 3 classes predicted take the 3 most
        # probable ones.
        elif np.sum(tmp_pred[i]) > 3:
            ind = np.argsort(probas[i])[-3:]
            tmp_pred[i] = np.zeros(28)
            tmp_pred[i, ind] = 1
    return(tmp_pred)

def get_pred_from_probas_threshold(probas, threshold=0.05):
    tmp_pred = np.greater(probas, threshold)
    for i in range(len(probas)):
        # if no classes predicted
        # i.e. no probas > 0.5
        # then choose the most probable one.
        if np.sum(tmp_pred[i]) == 0:
            try:
                tmp_pred[i, np.argmax(probas[i])[0]] = 1
            except IndexError:
                tmp_pred[i, np.argmax(probas[i])] = 1
    return(tmp_pred)


class Predictor:
    """ This class defines a Predictor object.
    It uses a loaded model to predict
    on the test images for Kaggle.
    """

    def __init__(self, sess, model, config):
        """ Init config, output file name for
        prediction.

        Args:
            sess: a tf session
            model: a loaded model (via model.load())
            config: a Bunch object
        """
        self.config = config
        self.sess = sess
        self.model = model
        # Defining the csv file name
        self.out_file = self.config.checkpoint_dir + 'prediction.csv'
        print("Writing to {}\n".format(self.out_file))

    def predict(self, testIterator):
        """ Uses a build model to
        predict one_hot_labels on the test set,
        these one_hot are then converted as required by
        Kaggle submission file.

        Args:
            testIterator: object of class DataTestLoader.
        """
        bin = MultiLabelBinarizer(classes=np.arange(28))
        bin.fit([[1]])  # needed for instantiation of the object
        predicted_labels = []
        # counter just for testing purpose
        # to stop the output after 8 predictions
        counter = 1
        for batch_imgs in testIterator.batch_iterator():
            if counter > 20:  # TODO remove
                break
            batch_probas = self.sess.run(self.model.out, {
                self.model.input: batch_imgs,
                self.model.is_training: False
            })
            #one_hot_batch_pred = get_pred_from_probas(batch_probas)
            one_hot_batch_pred = get_pred_from_probas_threshold(batch_probas)
            batch_pred = bin.inverse_transform(one_hot_batch_pred)
            predicted_labels = np.append(predicted_labels, [
                ' '.join([str(p) for p in sample_pred])
                for sample_pred in batch_pred
            ])
            counter += 1
            if counter % 1 == 0:
                print(counter*self.config.batch_size)
        ids = testIterator.image_ids[0:len(predicted_labels)]
        # print(np.shape(ids))
        result = pd.DataFrame()
        print(np.shape(predicted_labels))
        result['Id'] = ids
        result['Predict'] = predicted_labels
        result.to_csv(self.out_file, index=False)
