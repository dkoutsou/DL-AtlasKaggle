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
