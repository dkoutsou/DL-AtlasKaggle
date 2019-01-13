"""Based on the implementation here: https://github.com/okraus/DeepLoc"""
import tensorflow as tf
from base.base_model import BaseModel


class DeepLocModel(BaseModel):
    def __init__(self, config):
        super(DeepLocModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        super(DeepLocModel, self).init_build_model()
        x = tf.layers.conv2d(
                inputs=self.input_layer, kernel_size=(3, 3), filters=64,
                strides=(1, 1), name='conv_1')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_1')
        x = tf.nn.relu(x, name='act_1')
        x = tf.layers.conv2d(
                inputs=x, kernel_size=(3, 3), filters=64,
                strides=(1, 1), name='conv_2')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_2')
        x = tf.nn.relu(x, name='act_2')
        x = tf.layers.max_pooling2d(
                inputs=x, pool_size=(2, 2), strides=(2, 2), padding='SAME',
                name='pool_1')
        x = tf.layers.conv2d(
                inputs=x, kernel_size=(3, 3), filters=128,
                strides=(1, 1), name='conv_3')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_3')
        x = tf.nn.relu(x, name='act_3')
        x = tf.layers.conv2d(
                inputs=x, kernel_size=(3, 3), filters=128,
                strides=(1, 1), name='conv_4')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_4')
        x = tf.nn.relu(x, name='act_4')
        x = tf.layers.max_pooling2d(
                inputs=x, pool_size=(2, 2), strides=(2, 2), padding='SAME',
                name='pool_2')
        x = tf.layers.conv2d(
                inputs=x, kernel_size=(3, 3), filters=256,
                strides=(1, 1), name='conv_5')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_5')
        x = tf.nn.relu(x, name='act_5')
        x = tf.layers.conv2d(
                inputs=x, kernel_size=(3, 3), filters=256,
                strides=(1, 1), name='conv_6')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_6')
        x = tf.nn.relu(x, name='act_6')
        x = tf.layers.conv2d(
                inputs=x, kernel_size=(3, 3), filters=256,
                strides=(1, 1), name='conv_7')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_7')
        x = tf.nn.relu(x, name='act_7')
        x = tf.layers.conv2d(
                inputs=x, kernel_size=(3, 3), filters=256,
                strides=(1, 1), name='conv_8')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_8')
        x = tf.nn.relu(x, name='act_8')
        x = tf.layers.max_pooling2d(
                inputs=x, pool_size=(2, 2), strides=(2, 2), padding='SAME',
                name='pool_3')
        x = tf.layers.flatten(inputs=x, name='flatten')
        x = tf.layers.dense(inputs=x, units=512, name='fc_1')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_9')
        x = tf.nn.relu(x, name='act_9')
        x = tf.layers.dense(inputs=x, units=512, name='fc_2')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn_10')
        x = tf.nn.relu(x, name='act_10')
        self.logits = tf.layers.dense(x, units=28, name='logits')

        super(DeepLocModel, self).build_loss_output()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
