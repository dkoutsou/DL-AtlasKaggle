from data_loader.data_generator import DataGenerator, DataTestLoader
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import parmap
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MultiLabelBinarizer


def get_features_from_batch_images(img, r, p):
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


def extract_features(all_batches, config, train=True):
    """ Main features extraction function.

    Args:
        all_batches: batches iterator either from
                    DataGenerator class or from TestLoader
                    class. If TestLoader set train to False.
        config: config file
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
    p = config.patch_size
    r = 512 // p
    labels = np.empty(0)
    feats = np.empty(0)
    for counter, tmp in enumerate(all_batches):
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
                get_features_from_batch_images, batch_img, r, p, pm_pbar=True))
        print(time.time() - t1)
        labels = np.concatenate((labels,
                                 batch_label)) if labels.size else batch_label
        feats = np.concatenate((feats,
                                batch_feats)) if feats.size else batch_feats
    if train:
        return feats, labels
    else:
        return feats


def get_baseline_CV_score(feats, labels, estimator, scores=['f1_macro']):
    """ Calculate the cross-validation score
    of the baseline estimator

    Args:
        feats: input features to classifiers
        labels: one-hot multilabel representation
        scores: array of scoring function to assess by CV

    Returns:
        cv_scores: an array of scores objects
    """
    cv_scores = []
    for score in scores:
        cv_scores = np.append(
            cv_scores, cross_val_score(
                estimator, feats, labels, scoring=score))
    return cv_scores


def fit_predict(train_feats, train_labels, test_feats, estimator):
    """ Wrapper for the fit + predict pipeline.

    Args:
        train_feats: matrix[n_samples, n_feats] with training features
        train_labels: one-hot multilabel representation [n_samples, n_classes]
        estimator: object derived from Sklearn BaseEstimator

    Note:
        RF estimator can return None as class, don't know how Kaggle
        handles that. TODO: check this case.
    """
    bin = MultiLabelBinarizer(classes=np.arange(28))
    bin.fit(train_labels)  # needed for instantiation of the object
    estimator.fit(train_feats, train_labels)
    one_hot_pred = estimator.predict(test_feats)
    predicted_labels = bin.inverse_transform(one_hot_pred)
    return (predicted_labels)


def main():
    """ Main procedure: extract features,
    get RF cross-validation performance,
    fit and predict baseline and save csv for Kaggle.

    Note:
        Set the batch_size to some big number ~1000
        for better parmap performance.
    """
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception:
        print("missing or invalid arguments")
        exit(0)

    # create your data generator
    TrainingSet = DataGenerator(config)
    all_batches = TrainingSet.batch_iterator()
    # extract features
    train_feats, train_labels = extract_features(all_batches, config)
    print(np.sum(train_labels, axis=1))
    print(np.sum(train_labels))
    # get cv score
    estimator = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    # estimator = RidgeClassifierCV()
    cv_scores = get_baseline_CV_score(train_feats, train_labels, estimator)
    print(cv_scores)

    # Load Test Set
    TestSet = DataTestLoader(config)
    test_batches = TestSet.batch_iterator()

    # Fit and predict for Kaggle
    test_feats = extract_features(test_batches, config, train=False)
    print(np.shape(test_feats))
    prediction = fit_predict(train_feats, train_labels, test_feats, estimator)
    ids = TestSet.image_ids
    print(np.shape(ids))
    result = pd.DataFrame()

    string_pred = [
        ' '.join([str(p) for p in sample_pred]) for sample_pred in prediction
    ]
    print(np.shape(string_pred))
    result['Id'] = ids
    result['Predict'] = string_pred
    print(result)
    create_dirs([config.summary_dir])
    result.to_csv(config.summary_dir + '/prediction.csv', index=False)


if __name__ == '__main__':
    main()
