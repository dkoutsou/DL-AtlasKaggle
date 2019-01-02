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

    def build_loss_output(self):
        with tf.name_scope("output"):
            self.out = tf.nn.sigmoid(self.logits, name='out')
        with tf.name_scope("loss"):
            if self.config.focalLoss:
                print("Using focal loss")
                if self.config.use_weighted_loss:
                    print("weighted loss")
                    self.loss = tf.losses.compute_weighted_loss(
                        focal_loss(labels=self.label,
                                   logits=self.logits, gamma=2),
                        weights=self.class_weights)
                else:
                    print("not weighted loss")
                    self.loss = tf.reduce_mean(
                        focal_loss(labels=self.label,
                                   logits=self.logits,
                                   gamma=2))
            elif self.config.use_f1_loss:
                    self.loss = tf.reduce_mean(
                        f1_loss(y_true = self.label, y_pred = self.out))
            elif self.config.use_weighted_loss:
                try:
                    self.loss = tf.losses.compute_weighted_loss(
                        tf.nn.weighted_cross_entropy_with_logits(
                            targets=self.label, logits=self.logits,
                            pos_weight=self.config.pos_label_coeff),
                        weights=self.class_weights)
                except AttributeError:
                    print('WARN: pos_label_coeff not set using 1')
                    self.loss = tf.losses.compute_weighted_loss(
                        tf.nn.weighted_cross_entropy_with_logits(
                            targets=self.label, logits=self.logits,
                            pos_weight=1),
                        weights=self.class_weights)               
            else:
                try:
                    self.loss = tf.reduce_mean(
                        tf.nn.weighted_cross_entropy_with_logits(
                            targets=self.label, logits=self.logits,
                            pos_weight=self.config.pos_label_coeff))
                except AttributeError:
                    print('WARN: pos_label_coeff not set using 1')
                    self.loss = tf.reduce_mean(
                        tf.nn.weighted_cross_entropy_with_logits(
                            targets=self.label, logits=self.logits,
                            pos_weight=1))
            self.train_step = tf.train.AdamOptimizer(
                self.config.learning_rate).minimize(
                    self.loss, global_step=self.global_step_tensor)
