# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DenseNet implementation with TPU support.
Original paper: (https://arxiv.org/abs/1608.06993)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Learning hyperaparmeters
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def conv(image, filters, strides=1, kernel_size=3):
    """Convolution with default options from the densenet paper."""
    # Use initialization from https://arxiv.org/pdf/1502.01852.pdf

    return tf.layers.conv2d(
        inputs=image,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=tf.identity,
        use_bias=False,
        padding="same",
        kernel_initializer=tf.variance_scaling_initializer(),
    )


def dense_block(image, filters, is_training):
    """Standard BN+Relu+conv block for DenseNet."""
    image = tf.layers.batch_normalization(
        inputs=image,
        axis=-1,
        training=is_training,
        fused=True,
        center=True,
        scale=True,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
    )

    # Add bottleneck layer to optimize computation and reduce HBM space
    image = tf.nn.relu(image)
    image = conv(image, 4 * filters, strides=1, kernel_size=1)
    image = tf.layers.batch_normalization(
        inputs=image,
        axis=-1,
        training=is_training,
        fused=True,
        center=True,
        scale=True,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
    )

    image = tf.nn.relu(image)
    return conv(image, filters)


def transition_layer(image, filters, is_training):
    """Construct the transition layer with specified growth rate."""

    image = tf.layers.batch_normalization(
        inputs=image,
        axis=-1,
        training=is_training,
        fused=True,
        center=True,
        scale=True,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
    )
    image = tf.nn.relu(image)
    conv_img = conv(image, filters=filters, kernel_size=1)
    return tf.layers.average_pooling2d(
      conv_img, pool_size=2, strides=2, padding="same")


def _int_shape(layer):
    return layer.get_shape().as_list()
