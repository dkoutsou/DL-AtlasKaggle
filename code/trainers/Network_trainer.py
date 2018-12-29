from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


class NetworkTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(NetworkTrainer, self).__init__(sess, model, data, config, logger)
        try:
            if self.config.use_weighted_loss:
                pass
        except AttributeError:
            print('WARN: use_weighted_loss not set - using False')
            self.config.use_weighted_loss = False

    def train_epoch(self):
        self.data.set_batch_iterator(type='train')
        loop = tqdm(range(self.data.train_batches_per_epoch))
        losses = []
        train_preds = []
        train_true = []
        for _ in loop:
            loss, pred, true_label = self.train_step()
            losses.append(loss)
            train_preds = np.append(train_preds, pred)
            train_true = np.append(train_true, true_label)
            cur_it = self.model.global_step_tensor.eval(self.sess)
            if cur_it % 10 == 0:
                # Save the training values every 10 steps
                train_loss = np.mean(losses)
                # i am not calculating f1 each steps in train_step()
                # because if there are no true label of one class
                # the metric is ill-defined and set to 0
                # but on 10 training batch the ill-defined case
                # nearly never happens leading to a meaningful f1-score.
                train_f1 = f1_score(train_true, train_preds, average='macro')
                losses = []
                f1s = []
                print('Step {}: training_loss:{}, training_f1:{}'.format(
                    cur_it, train_loss, train_f1))
                train_summaries_dict = {
                    'loss': train_loss,
                    'f1': train_f1
                }
                self.logger.summarize(
                    cur_it, summaries_dict=train_summaries_dict)

            if (cur_it % 100 == 0) and (cur_it > 0):
                # Evaluate on val every epoch
                val_loss, val_f1 = self.val_step()
                print('Step {}: val_loss:{}, val_f1:{}'.format(
                    cur_it, val_loss, val_f1))
                val_summaries_dict = {'loss': val_loss, 'f1': val_f1}
                self.logger.summarize(
                    cur_it, summaries_dict=val_summaries_dict,
                    summarizer='test')

            if (cur_it % 200 == 0) and (cur_it > 0):
                self.model.save(self.sess)

        train_loss = np.mean(losses)
        train_f1 = np.mean(f1s)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        print('Step {}: training_loss:{}, training_f1:{}'.format(
            cur_it, train_loss, train_f1))
        train_summaries_dict = {
            'loss': train_loss,
            'f1': train_f1
        }
        # Evaluate on val every epoch
        val_loss, val_f1 = self.val_step()
        print('Step {}: val_loss:{}, val_f1:{}'.format(
            cur_it, val_loss, val_f1))
        val_summaries_dict = {'loss': val_loss, 'f1': val_f1}
        self.logger.summarize(cur_it, summaries_dict=train_summaries_dict)
        self.logger.summarize(
            cur_it, summaries_dict=val_summaries_dict, summarizer='test')
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.train_iterator)
        print(np.shape(batch_y))
        feed_dict = {
            self.model.input: batch_x,
            self.model.label: batch_y,
            self.model.is_training: True,
            self.model.class_weights: self.data.class_weights
        }
        _, loss, pred = self.sess.run([
            self.model.train_step, self.model.loss,
            self.model.prediction
        ],
            feed_dict=feed_dict)
        return loss, pred, batch_y

    def val_step(self):
        val_iterator = self.data.batch_iterator(type='val')
        val_losses = []
        val_preds = []
        val_true = []
        for batch_x, batch_y in val_iterator:
            feed_dict = {
                self.model.input: batch_x,
                self.model.label: batch_y,
                self.model.is_training: False,
                self.model.class_weights: self.data.class_weights
            }
            loss, pred = self.sess.run(
                [self.model.loss, self.model.prediction],
                feed_dict=feed_dict)
            val_losses.append(loss)
            val_preds = np.append(val_preds, pred)
            val_true = np.append(val_true, batch_y)
        val_f1 = f1_score(val_true, val_preds, average='macro')
        val_loss = np.mean(val_losses)
        return val_loss, val_f1
