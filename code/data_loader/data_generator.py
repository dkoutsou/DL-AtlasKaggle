import numpy as np
import tensorflow as tf
import os 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class DataGenerator:
    def __init__(self, config):
        cwd = os.getcwd()
        self.config = config
        # Read csv file
        tmp = pd.read_csv(os.path.abspath(cwd + self.config.data_path + 'train.csv'), delimiter=',')
        # A vector of images id.
        image_ids = tmp["Id"]
        self.n = len(image_ids)
        # for each id sublist of the 4 filenames [batch_size, 4]
        self.filenames = np.asarray([[cwd + self.config.data_path + 'train/'+ id + '_' + c +'.png'for c in ['red', 'green', 'yellow', 'blue']] for id in image_ids])
        # Labels 
        self.labels = np.reshape(tmp["Target"],(-1,1))

    def next_batch(self):        
        idx = np.random.choice(self.n, self.config.batch_size)
        print(idx)
        batchfile, batchlabel = self.filenames[idx], self.labels[idx]
        batchimages = np.asarray([[np.asarray(Image.open(x)) for x in y] for y in batchfile])
        yield batchimages, batchlabel



if __name__ == '__main__':
    # just for testing
    from bunch import Bunch
    config_dict = {'data_path':'/data/', 'batch_size': 32}
    config = Bunch(config_dict)
    TrainingSet = DataGenerator(config)
    batch = TrainingSet.next_batch()
    for img, y in batch:
        print(np.shape(img))
        print(np.shape(y))
    



