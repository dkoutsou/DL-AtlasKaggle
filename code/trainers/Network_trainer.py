from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from utils.predictor import get_pred_from_probas


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
        train_probas = []
        train_true = []
        for _ in loop:
            loss, pred, true_label = self.train_step()
            losses.append(loss)
            train_probas = np.append(train_probas, pred)
            train_true = np.append(train_true, true_label)
            cur_it = self.model.global_step_tensor.eval(self.sess)
            print(loss)
            if cur_it % (self.data.train_batches_per_epoch//5) == 0:
                print(train_true[0:28])
                print(train_probas[0:28])
                print(loss)
                # Save the training values every 10 steps
                train_loss = np.mean(losses)
                train_true = np.reshape(train_true, (-1, 28))
                train_probas = np.reshape(train_probas, (-1, 28))
                train_f1 = f1_score(train_true, get_pred_from_probas(
                    train_probas), average='macro')
                train_f1_2 = f1_score(train_true, np.greater(
                    train_probas, 0.05), average='macro')
                train_f1_3 = f1_score(train_true, np.greater(
                    train_probas, 0.1), average='macro')
                train_f1_4 = f1_score(train_true, np.greater(
                    train_probas, 0.2), average='macro')
                losses = []
                train_probas = []
                train_true = []
                print(
                    'Step {}: training_loss:{}, f1:{}'
                    ', f1_005:{}, f1_01:{}, f1_02:{}'
                    .format(
                        cur_it, train_loss, train_f1, train_f1_2,
                        train_f1_3, train_f1_4))
                train_summaries_dict = {
                    'loss': train_loss,
                    'f1': train_f1,
                    'f1_005_thres': train_f1_2,
                    'f1_01_thres': train_f1_3,
                    'f1_02_thres': train_f1_4,
                }
                self.logger.summarize(
                    cur_it, summaries_dict=train_summaries_dict) 
        # Saving every epoch
        self.model.save(self.sess)
        # Evaluate on validation at the end of every epoch
        val_loss, val_f1, \
            val_f1_2, val_f1_3, val_f1_4 = self.val_step()
        print('Step {}: val_loss:{}, val_f1:{},'
                ' val_f1_005:{}, val_f1_01:{}, val_f1_02:{} '
                .format(
                    cur_it, val_loss, val_f1, val_f1_2,
                    val_f1_3, val_f1_4))
        val_summaries_dict = {'loss': val_loss,
                                'f1': val_f1,
                                'f1_005_thres': val_f1_2,
                                'f1_01_thres': val_f1_3,
                                'f1_02_thres': val_f1_4}
        self.logger.summarize(
            cur_it, summaries_dict=val_summaries_dict,
            summarizer='test')

    def train_step(self):
        batch_x, batch_y = next(self.data.train_iterator)
        print(np.shape(batch_y))
        feed_dict = {
            self.model.input: batch_x,
            self.model.label: batch_y,
            self.model.is_training: True,
            self.model.class_weights: self.data.class_weights
        }
        _, loss, out = self.sess.run([
            self.model.train_step, self.model.loss,
            self.model.out
        ],
            feed_dict=feed_dict)
        return loss, out, batch_y

    def val_step(self):
        val_iterator = self.data.batch_iterator(type='val')
        val_losses = []
        val_probas = []
        val_true = []
        for batch_x, batch_y in val_iterator:
            feed_dict = {
                self.model.input: batch_x,
                self.model.label: batch_y,
                self.model.is_training: False,
                self.model.class_weights: self.data.class_weights
            }
            loss, out = self.sess.run(
                [self.model.loss, self.model.out],
                feed_dict=feed_dict)
            val_losses.append(loss)
            val_probas = np.append(val_probas, out)
            val_true = np.append(val_true, batch_y)
        val_true = np.reshape(val_true, (-1, 28))
        val_probas = np.reshape(val_probas, (-1, 28))
        val_preds = get_pred_from_probas(val_probas)
        val_f1 = f1_score(val_true, val_preds, average='macro')
        val_f1_2 = f1_score(val_true, np.greater(
            val_probas, 0.05), average='macro')
        val_f1_3 = f1_score(val_true, np.greater(
            val_probas, 0.1), average='macro')
        val_f1_4 = f1_score(val_true, np.greater(
            val_probas, 0.2), average='macro')
        val_loss = np.mean(val_losses)
        return val_loss, val_f1, val_f1_2, val_f1_3, val_f1_4
