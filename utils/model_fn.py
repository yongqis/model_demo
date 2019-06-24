"""Define the model."""

import tensorflow as tf

from utils.triplet_loss import batch_all_triplet_loss
from utils.triplet_loss import batch_hard_triplet_loss
from slim.nets import vgg
slim = tf.contrib.slim


def vgg_model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    images = features
    labels = tf.cast(labels, tf.int64)
    # images = tf.reshape(images, [-1, params.image_size, params.image_size, 1])
    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)

    # MODEL: define the layers of the model
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    embeddings = vgg.vgg_16(
        inputs=images,
        num_classes=params.embedding_size,
        is_training=is_training,
        dropout_keep_prob=params.dropout_keep_prob)
    # 增加一个l2 norm
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin, squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin, squared=params.squared)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()

    # Add a dependency to update the moving mean and variance for batch normalization
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def se_vgg_model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    images = features
    labels = tf.cast(labels, tf.int64)
    # images = tf.reshape(images, [-1, params.image_size, params.image_size, 1])
    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)

    # MODEL: define the layers of the model
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    _, end_points = vgg.vgg_16(
        images,
        num_classes=1000,
        is_training=is_training,
        dropout_keep_prob=0.5,
        spatial_squeeze=True,
        scope='vgg_16',
        fc_conv_padding='VALID',
        global_pool=False)
    # SE module 2 loc feature vector
    net = end_points['pool5']
    loc_feature1 = se_moduel(net, 16)
    loc_feature2 = se_moduel(net, 16)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()

    # Add a dependency to update the moving mean and variance for batch normalization
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def se_moduel(feature_map, r):
    """

    :param feature_map:
    :param r:
    :return:
    """
    feature = tf.reduce_mean(feature_map, [1, 2], keepdims=True)
    feature = slim.conv2d(feature, 4096 / r, [1, 1], strides=1, padding='VALID', activation_fn=tf.nn.relu6)
    feature = slim.conv2d(feature, 4096, [1, 1], stride=1, padding='VALID', activation_fn=tf.nn.sigmoid)
    feature_map = tf.multiply(feature_map, feature)  #
    return feature_map
