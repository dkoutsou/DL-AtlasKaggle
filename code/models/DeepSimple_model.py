import tensorflow as tf
from base.base_model import BaseModel


class DeepSimpleModel(BaseModel):
    def __init__(self, config):
        super(DeepSimpleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        super(DeepSimpleModel, self).init_build_model()
        print(self.input_layer.get_shape())
        # Block 1
        x = tf.layers.dropout(self.input_layer, rate=0.1,
                              training=self.is_training)
        x = tf.layers.conv2d(x, 8, 3,
                             padding='same')
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.nn.relu(x, name='act1')
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        print(x.get_shape())
        x = tf.layers.conv2d(x, 16, 5, padding='same')
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2))
        print(x.get_shape())
        # Block 2
        x = tf.layers.conv2d(x, 32, 3, padding='same')
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        x = tf.layers.conv2d(x, 32, 5, padding='same')
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2))
        print(x.get_shape())
        # Block 3
        x = tf.layers.conv2d(x, 64, 3, padding='same')
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, 256, 3, padding='same')
        x = tf.layers.batch_normalization(
            x, training=self.is_training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2))
        print(x.get_shape())
        # Classification block
        x = tf.layers.flatten(x, name='flatten')
        x = tf.nn.relu(x)
        self.logits = tf.layers.dense(x, units=28, name='logits')

        super(DeepSimpleModel, self).build_loss_output()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
