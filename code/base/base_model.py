import tensorflow as tf
from utils.loss import focal_loss, f1_loss


class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function that saves the checkpoint
    # in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir,
                        self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined
    # in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(
            self.config.checkpoint_dir)
        # latest_checkpoint = self.config.checkpoint_dir + '-5600'
        # self.saver = tf.train.import_meta_graph("{}.meta"
        #                                         .format(latest_checkpoint))
        if latest_checkpoint:
            print(
                "Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(
                0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(
                self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(
                0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def init_build_model(self):
        try:
            if self.config.input_size:
                pass
        except AttributeError:
            print('WARN: input_size not set - using 512')
            self.config.input_size = 512
        self.is_training = tf.placeholder(tf.bool)
        self.class_weights = tf.placeholder(
            tf.float32, shape=[1, 28], name="weights")
        self.class_weights = tf.stop_gradient(self.class_weights,
                                              name="stop_gradient")
        self.input = tf.placeholder(
            tf.float32, shape=[None, 4, 512, 512], name="input")
        self.label = tf.placeholder(tf.float32, shape=[None, 28])
        x = tf.transpose(self.input, perm=[0, 2, 3, 1])
        self.input_layer = tf.image.resize_images(x, (self.config.input_size,
                                                      self.config.input_size))

    def build_loss_output(self):
        try:
            if self.config.use_weighted_loss:
                pass
        except AttributeError:
            print('WARN: use_weighted_loss not set - using False')
            self.config.use_weighted_loss = False
        try:
            if self.config.focalLoss:
                pass
        except AttributeError:
            print('WARN: focalLoss not set - using False')
            self.config.focalLoss = False
        try:
            if self.config.f1_loss:
                pass
        except AttributeError:
            print('WARN: f1_loss not set - using False')
            self.config.f1_loss = False
        with tf.name_scope("output"):
            self.out = tf.nn.sigmoid(self.logits, name='out')
        with tf.name_scope("loss"):
            if self.config.focalLoss:
                print("Using focal loss")
                self.loss = tf.reduce_mean(
                    focal_loss(labels=self.label,
                               logits=self.logits,
                               gamma=2))
            elif self.config.f1_loss:
                self.loss = f1_loss(y_true=self.label,
                                    y_pred=self.out)
            elif self.config.use_weighted_loss:
                self.loss = tf.losses.compute_weighted_loss(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label, logits=self.logits),
                    weights=self.class_weights)
            else:
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label, logits=self.logits))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(
                    self.config.learning_rate).minimize(
                        self.loss, global_step=self.global_step_tensor)
