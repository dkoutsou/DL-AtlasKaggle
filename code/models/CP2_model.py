import tensorflow as tf
from base.base_model import BaseModel


class CP2Model(BaseModel):
    def __init__(self, config):
        super(CP2Model, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        try:
            if self.config.use_weighted_loss:
                pass
        except AttributeError:
            print('WARN: use_weighted_loss not set - using False')
            self.config.use_weighted_loss = False
        self.is_training = tf.placeholder(tf.bool)
        self.class_weights = tf.placeholder(
            tf.float32, shape=[1, 28], name="weights")
        self.class_weights = tf.stop_gradient(self.class_weights,
                                              name="stop_gradient")
        self.input = tf.placeholder(
            tf.float32, shape=[None, 4, 512, 512], name="input")
        self.label = tf.placeholder(tf.float32, shape=[None, 28])

        # All tf functions work better with channel first
        # otherwise some fail on CPU (known issue)
        x = tf.transpose(self.input, perm=[0, 2, 3, 1])
        x = tf.image.resize_images(x, (256,256))
        # Block 1
        x = tf.layers.conv2d(x, 64, 3, padding='same', name='conv1_1')
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
