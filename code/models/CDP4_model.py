import tensorflow as tf
from base.base_model import BaseModel


class CDP4Model(BaseModel):
    def __init__(self, config):
        super(CDP4Model, self).__init__(config)
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
        self.input = tf.placeholder(
            tf.float32, shape=[None, 4, 512, 512], name="input")
        self.label = tf.placeholder(tf.float32, shape=[None, 28])

        # All tf functions work better with channel first
        # otherwise some fail on CPU (known issue)
        x = tf.transpose(self.input, perm=[0, 2, 3, 1])
        print(x.get_shape())
        # Block 1
        x = tf.layers.conv2d(x, 64, 3, padding='same', name='conv1')
        x = tf.nn.relu(x, name='act1')
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool1')
        print(x.get_shape())
        # Block 2
        x = tf.layers.conv2d(x, 128, 3, padding='same', name='conv2')
        x = tf.nn.relu(x, name='act2')
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool2')
        print(x.get_shape())
        # Block 3
        x = tf.layers.conv2d(x, 256, 3, padding='same', name='conv3')
        x = tf.nn.relu(x, name='act3')
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        print(x.get_shape())
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool3')
        # Block 4
        x = tf.layers.conv2d(x, 256, 3, padding='same', name='conv4')
        x = tf.nn.relu(x, name='act4')
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool4')
        # Classification block
        x = tf.layers.flatten(x, name='flatten')
        x = tf.nn.relu(x, name='act5')
        logits = tf.layers.dense(x, units=28, name='logits')
        out = tf.nn.sigmoid(logits, name='out')
        with tf.name_scope("loss"):
            if self.config.use_weighted_loss:
                tf.stop_gradient(self.class_weights, name="stop_gradient")
                self.loss = tf.losses.compute_weighted_loss(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label, logits=logits),
                    weights=self.class_weights)
            else:
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label, logits=logits))
            self.train_step = tf.train.AdamOptimizer(
                self.config.learning_rate).minimize(
                    self.loss, global_step=self.global_step_tensor)
        with tf.name_scope("output"):
            self.prediction = tf.round(out, name="prediction")

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
