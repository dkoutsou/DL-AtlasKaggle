import tensorflow as tf


def f1_loss(y_true, y_pred):
    # y_pred should be the proba to make it differentiable
    tp = tf.reduce_sum(tf.cast(y_true*y_pred, 'float'), axis=0)
    # tn = tf.reduce_sum(tf.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2*p*r / (p+r+tf.keras.backend.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - tf.reduce_mean(f1)


def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Based on https://www.kaggle.com/achoetwice/focal-loss-with-pre-train-v2
    with modifications from Keras to TF

    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)

        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise

        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise

        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """

    # Define epsilon so that the backpropagation will not result in NaN
    # for 0 divisor case
    epsilon = 1e-7
    # Add the epsilon to prediction value
    # y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0-epsilon)
    # Calculate p_t
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
    # Calculate alpha_t
    alpha_factor = tf.ones_like(y_true)*alpha
    alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1-alpha_factor)
    # Calculate cross entropy
    cross_entropy = -tf.log(p_t)
    weight = alpha_t * tf.pow((1-p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = tf.reduce_sum(loss, axis=1)

    return loss
