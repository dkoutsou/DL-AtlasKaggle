import tensorflow as tf
from base.base_model import BaseModel


class KaggleModel(BaseModel):
    def __init__(self, config):
        super(KaggleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        super(KaggleModel, self).init_build_model()
        # Block 1
        x = tf.layers.batch_normalization(
            self.input_layer, training=self.is_training)
        x = tf.layers.conv2d(self.input_layer, 8, 3,
                             padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.layers.conv2d(x, 8, 3,
                             padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.layers.conv2d(x, 16, 3,
                             padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2))
        x = tf.layers.dropout(x, rate=0.25,
                              training=self.is_training)
        c1 = tf.layers.conv2d(x, 16, 3,
                              padding='same')
        c1 = tf.nn.relu(c1)
        c2 = tf.layers.conv2d(x, 16, 5,
                              padding='same')
        c2 = tf.nn.relu(c2)
        c3 = tf.layers.conv2d(x, 16, 7,
                              padding='same')
        c3 = tf.nn.relu(c3)
        c4 = tf.layers.conv2d(x, 16, 1,
                              padding='same')
        c4 = tf.nn.relu(c4)
        x = tf.concat([c1, c2, c3, c4], 0)
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2))
        x = tf.layers.dropout(x, rate=0.25,
                              training=self.is_training)
        x = tf.layers.conv2d(x, 32, 3,
                             padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2))
        x = tf.layers.dropout(x, rate=0.25,
                              training=self.is_training)
        x = tf.layers.conv2d(x, 128, 3,
                             padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2))
        x = tf.layers.dropout(x, rate=0.25,
                              training=self.is_training)
        # Classification
        x = tf.layers.flatten(x, name='flatten')
        x = tf.nn.relu(x, name='act5')
        self.logits = tf.layers.dense(x, units=28, name='logits')

        super(KaggleModel, self).build_loss_output()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
