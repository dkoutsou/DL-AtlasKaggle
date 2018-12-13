import numpy as np
import tensorflow as tf
import os 
import pandas as pd
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self, config):
        cwd = os.getcwd()
        self.config = config
        # Read csv file
        tmp = pd.read_csv(os.path.abspath(cwd + self.config.data_path + 'train.csv'), delimiter=',')
        # A vector of images id.
        image_ids = tmp["Id"]
        # for each id sublist of the 4 filenames
        four_filenames_list = [[cwd + self.config.data_path + 'train/'+ id + '_' + c +'.png'for c in ['red', 'green', 'yellow', 'blue']] for id in image_ids[0:1]]
        print(four_filenames_list)
        # put every thing in one tensor of dimension [batch_size, 512, 512, 4]
        print(tf.shape(tf.image.decode_jpeg(four_filenames_list[0][0])))
        four_images = [[tf.expand_dims(tf.image.decode_jpeg(x), 2) for x in y] for y in four_filenames_list]
        four_channel_input = tf.concat(four_images, axis=2)
        # Labels 
        labels = tmp["Target"]
        # Creating the dataset and prepare for batches
        self.data = tf.data.Dataset.from_tensor_slices((four_channel_input, labels))
        # Shuffle and batch
        self.data = self.data.shuffle(buffer_size=10000)
        self.data = self.data.batch(self.config.batch_size)
        self.data_iterator = self.data.make_one_shot_iterator()
    
    def next_batch(self):        
        return self.data_iterator.get_next()


if __name__ == '__main__':
    # just for testing
    from bunch import Bunch
    config_dict = {'data_path':'/data/', 'batch_size': 32}
    config = Bunch(config_dict)
    TrainingSet = DataGenerator(config)
    sess = tf.Session()
    batch = TrainingSet.next_batch()
    print(sess.run(batch))
    



