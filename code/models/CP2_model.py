import tensorflow as tf
from base.base_model import BaseModel


class CP2Model(BaseModel):
    def __init__(self, config):
        super(CP2Model, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):

        super(CP2Model, self).init_build_model()
        # Block 1
        x = tf.layers.conv2d(self.input_layer, 64, 3,
                             padding='same', name='conv1_1')
        x = tf.nn.relu(x, name='act1_1')
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool1')
        # Block 2
        x = tf.layers.conv2d(x, 128, 3, padding='same', name='conv2_1')
        x = tf.nn.relu(x, name='act2_1')
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool2')
        # Classification block
        x = tf.layers.flatten(x, name='flatten')
        x = tf.nn.relu(x, name='act4')
        self.logits = tf.layers.dense(x, units=28, name='logits')

        super(CP2Model, self).build_loss_output()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
