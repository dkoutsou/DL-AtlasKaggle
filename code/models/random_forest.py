import time

import numpy as np
import parmap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class RandomForestBaseline(BaseEstimator, TransformerMixin):
    """
    Wrapper around RandomForestClassifier with some additional rules
    """

    def __init__(self,
                 n_estimators=1000,
                 n_jobs=None,
                 random_state=None,
                 class_weight=None):
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.class_weight = class_weight
        self.rf = None

    def _get_features_from_batch_images(self, img, r, p):
        """ Get the features from one image.
        Global intensity histogram per channel
        and local histogram on image patches.

        Args:
            img: the image with 4 input channel
            r: 512//p number of patches in
                one side of the image
            p: patch size for local histogram
        """
        tmp_feats = []
        for channel in range(4):
            current_img = img[channel, :, :]
            tmp_feats = np.append(tmp_feats, np.histogram(current_img)[0])
            # extract 8*8 patches of 64*64 px and derive 10 bins histogram
            for j in range(r):
                for k in range(r):
                    tmp_feats = np.append(
                        tmp_feats,
                        np.histogram(current_img[j * p:(j + 1) * (p), k *
                                                 p:(k + 1) * p])[0])
        return tmp_feats

    def _extract_features(self, all_batches, patch_size, train=True):
        """ Main features extraction function.

        Args:
            all_batches: batches iterator either from
                        DataGenerator class or from TestLoader
                        class. If TestLoader set train to False.
            patch_size: patch_size to extract features
            train: boolean to specify with DataLoader is used and
                    whether should return labels or not.
        Returns:
            if train:
                feats: [n_samples, n_feats]
                labels: [n_samples, 1]
            else:
                feats: [n_samples, n_feats]
        """
        # manually derive basic intensities features
        # takes 20 sec / 1048 images batch on my laptop in 4 cores //
        p = patch_size
        r = 512 // p
        labels = np.empty(0)
        feats = np.empty(0)
        for counter, tmp in enumerate(all_batches):
            # if counter == 2:
            # break
            if train:
                batch_img, batch_label = tmp
            else:
                batch_img = tmp
                batch_label = np.empty(0)
            # just for testing just use 20 batch as training set
            print('processing batch {}'.format(counter))
            t1 = time.time()
            batch_feats = np.asarray(
                parmap.map(
                    self._get_features_from_batch_images,
                    batch_img,
                    r,
                    p,
                    pm_pbar=True))
            print(time.time() - t1)
            labels = np.concatenate(
                (labels, batch_label)) if labels.size else batch_label
            feats = np.concatenate(
                (feats, batch_feats)) if feats.size else batch_feats
        if train:
            return feats, labels
        else:
            return feats

    def _proba_2_one_hot(self, proba):
        tmp = proba > 0.5
        all_false = np.all(~tmp)
        # If no class has higher than 0.5 probability
        if all_false:
            max_prob = np.max(proba)

            # Pick top 3 classes
            prediction = np.zeros(proba.shape, dtype=np.int)
            inds = np.argpartition(proba, -3)[-3:]
            prediction[inds] = 1

            # From the top 3 discard those that do not reach the threshold
            inds = proba < (0.90 * max_prob)
            prediction[inds] = 0
            return prediction
        else:
            # encode classes that have higher than 0.5
            return tmp.astype(int)

    def fit(self, X, y, sample_weight=None):
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            class_weight=self.class_weight)
        return self.rf.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        probas = self.rf.predict_proba(X)
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        # Create (n_sample * n_classes) matrix wtih probabilities
        probas = [class_probs[:, 1].reshape(-1, 1) for class_probs in probas]
        probas = np.hstack(probas)

        for i in range(probas.shape[0]):
            probas[i, :] = self._proba_2_one_hot(probas[i, :])

        probas = probas.astype(np.int)
        return probas
