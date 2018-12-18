import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import re


class DataGenerator:
    """A class that implements an iterator to load the data. It uses  as an
    environmental variable the data folder and then loads the necessary files
    (labels and images) and starts loading the data
    """

    def __init__(self, config):
        """The constructor of the DataGenerator class. It loads the training
        labels and the images.

        Parameters
        ----------
            config: dict
                a dictionary with necessary information for the dataloader
                (e.g batch size)
        """
        cwd = os.getenv("DATA_PATH")
        if cwd is None:
            print("Set your DATA_PATH env first")
            sys.exit(1)
        self.config = config
        # Read csv file
        tmp = pd.read_csv(os.path.abspath(cwd + 'train.csv'), delimiter=',')
        # A vector of images id.
        image_ids = tmp["Id"]
        self.n = len(image_ids)
        # for each id sublist of the 4 filenames [batch_size, 4]
        self.filenames = np.asarray(
            [[cwd + '/train/' + id + '_' + c + '.png'
              for c in ['red', 'green', 'yellow', 'blue']]
             for id in image_ids])
        # Labels
        self.labels = tmp["Target"].values
        # Number batches per epoch
        self.num_batches_per_epoch = int((self.n-1)/self.config.batch_size) + 1
    
    def batch_iterator(self):
        """
        Generates a batch iterator for the dataset.
        """
        binarizer = MultiLabelBinarizer(classes=np.arange(28))
        # use 1 as default if num_epochs is not specified (i.e. for baseline)
        try:
            r = self.config.num_epochs
        except AttributeError:
            print('WARN: num_epochs not set - using 1')
            r = 1
        for _ in range(r):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(self.n))
            shuffled_filenames = self.filenames[shuffle_indices]
            shuffled_labels = self.labels[shuffle_indices]
            for batch_num in range(self.num_batches_per_epoch):
                start_index = batch_num * self.config.batch_size
                end_index = min((batch_num + 1) *
                                self.config.batch_size, self.n)
                batchfile = shuffled_filenames[start_index:end_index]
                batchlabel = shuffled_labels[start_index:end_index]
                # To one-hot representation of labels
                # e.g. before e.g. ['22 0' '12 23 0']
                # after split [['22', '0'], ['12', '23', '0']]
                # after binarize it is one hot representation
                batchlabel = [[int(c) for c in l.split(' ')]
                              for l in batchlabel]
                batchlabel = binarizer.fit_transform(batchlabel)
                batchimages = np.asarray(
                    [[np.asarray(Image.open(x)) for x in y]
                     for y in batchfile])
                yield batchimages, batchlabel


class DataTestLoader:
    """A class that implements an iterator to load the data. It uses  as an
    environmental variable the data folder and then loads the necessary files
    (labels and images) and starts loading the data
    """

    def __init__(self, config):
        """The constructor of the DataTestLoader class. It loads the testing images.

        Parameters
        ----------
            config: dict
                a dictionary with necessary information for the dataloader
                (e.g batch size)
        """
        cwd = os.getenv("DATA_PATH")
        if cwd is None:
            print("Set your DATA_PATH env first")
            sys.exit(1)
        self.config = config
        list_files = [f for f in os.listdir(cwd + '/test/')]
        self.image_ids = list(set([re.search(
            '(?P<word>[\w|-]+)\_[a-z]+.png', s).group('word')
             for s in list_files]))
        self.n = len(self.image_ids)
        # for each id sublist of the 4 filenames [batch_size, 4]
        self.filenames = np.asarray(
            [[cwd + 'test/' + id + '_' + c + '.png'
              for c in ['red', 'green', 'yellow', 'blue']]
             for id in self.image_ids])

    def batch_iterator(self):
        """
        Generates a batch iterator for the dataset.
        """
        num_batches_per_epoch = int((self.n-1)/self.config.batch_size) + 1
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * self.config.batch_size
            end_index = min((batch_num + 1) *
                            self.config.batch_size, self.n)
            batchfile = self.filenames[start_index:end_index]
            batchimages = np.asarray(
                [[np.asarray(Image.open(x)) for x in y]
                    for y in batchfile])
            yield batchimages


if __name__ == '__main__':
    # just for testing
    from bunch import Bunch
    config_dict = {'batch_size': 32}
    config = Bunch(config_dict)
    TrainingSet = DataGenerator(config)
    """
    all_batches = TrainingSet.batch_iterator()
    for batch_x, batch_y in all_batches:
        print(np.shape(batch_x))   # (32, 4, 512, 512)
        print(np.shape(batch_y))
    """
    TestLoader = DataTestLoader(config)
    all_test_batches = TestLoader.batch_iterator()
    for batch_x in all_test_batches:
        print(np.shape(batch_x))   # (32, 4, 512, 512)
