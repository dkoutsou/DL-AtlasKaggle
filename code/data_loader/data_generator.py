import numpy as np
import os
import sys
import pandas as pd
from PIL import Image


class DataGenerator:
    def __init__(self, config):
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
        self.labels = tmp["Target"].values.reshape((-1, 1))

    def next_batch(self):
        idx = np.random.choice(self.n, self.config.batch_size)
        batchfile, batchlabel = self.filenames[idx], self.labels[idx]
        batchimages = np.asarray([[np.asarray(Image.open(x)) for x in y]
                                  for y in batchfile])
        yield batchimages, batchlabel


if __name__ == '__main__':
    # just for testing
    from bunch import Bunch
    config_dict = {'batch_size': 32}
    config = Bunch(config_dict)
    TrainingSet = DataGenerator(config)
    batch = TrainingSet.next_batch()
    for img, y in batch:
        print(np.shape(img))
        print(np.shape(y))
