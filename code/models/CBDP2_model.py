import tensorflow as tf
from base.base_model import BaseModel


class CBDP2Model(BaseModel):
    def __init__(self, config):
        super(CBDP2Model, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        super(CBDP2Model, self).init_build_model()
        # Block 1
        x = tf.layers.conv2d(self.input_layer, 32, 3,
                             padding='same', name='conv1')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn1')
        x = tf.nn.relu(x, name='act1')
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool1')
        # Block 2
        x = tf.layers.conv2d(x, 64, 3, padding='same', name='conv2')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn2')
        x = tf.nn.relu(x, name='act2')
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool2')
        # Classification block
        x = tf.layers.flatten(x, name='flatten')
        x = tf.nn.relu(x, name='act5')
        self.logits = tf.layers.dense(x, units=28, name='logits')

        super(CBDP2Model, self).build_loss_output()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
