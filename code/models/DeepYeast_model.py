import tensorflow as tf
from base.base_model import BaseModel


class DeepYeastModel(BaseModel):
    def __init__(self, config):
        super(DeepYeastModel, self).__init__(config)
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
        # Architecture translated from DeepYeast source code
        # original code was written in tf.keras

        # All tf functions work better with channel first
        # otherwise some fail on CPU (known issue)
        x = tf.transpose(self.input, perm=[0, 2, 3, 1])
        # Block 1
        x = tf.layers.conv2d(x, 64, 3, padding='same', name='conv1_1')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn1_1')
        x = tf.nn.relu(x, name='act1_1')
        x = tf.layers.conv2d(x, 64, 3, padding='same', name='conv1_2')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn1_2')
        x = tf.nn.relu(x, name='act1_2')
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool1')
        # Block 2
        x = tf.layers.conv2d(x, 128, 3, padding='same', name='conv2_1')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn2_1')
        x = tf.nn.relu(x, name='act2_1')
        x = tf.layers.conv2d(x, 128, 3, padding='same', name='conv2_2')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn2_2')
        x = tf.nn.relu(x, name='act2_2')
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool2')
        # Block 3
        x = tf.layers.conv2d(x, 256, 3, padding='same', name='conv3_1')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn3_1')
        x = tf.nn.relu(x, name='act3_1')
        x = tf.layers.conv2d(x, 256, 3, padding='same', name='conv3_2')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn3_2')
        x = tf.nn.relu(x, name='act3_2')
        x = tf.layers.conv2d(x, 256, 3, padding='same', name='conv3_3')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn3_3')
        x = tf.nn.relu(x, name='act3_3')
        x = tf.layers.conv2d(x, 256, 3, padding='same', name='conv3_4')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn3_4')
        x = tf.nn.relu(x, name='act3_4')
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool3')
        # Classification block
        x = tf.layers.flatten(x, name='flatten')
        x = tf.layers.batch_normalization(
            x, training=self.is_training, name='bn4')
        x = tf.nn.relu(x, name='act4')
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training)
        logits = tf.layers.dense(x, units=28, name='logits')
        # we have to adapt their code cause their code does
        # one label prediction, we want multilabel
        # use sigmoid not softmax because multilabel
        # then each out node is the proba the corresponding
        # label being true. I.e. if > 0.5 output the prediction.
        with tf.name_scope("loss"):
            if self.config.use_weighted_loss:
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
            self.out = tf.nn.sigmoid(logits, name='out')
            self.prediction = tf.round(self.out, name="prediction")

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
