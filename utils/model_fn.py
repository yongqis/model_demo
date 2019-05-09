"""Define the model."""

import tensorflow as tf

from model_demo.utils.triplet_loss import batch_all_triplet_loss
from model_demo.utils.triplet_loss import batch_hard_triplet_loss
from model_demo.utils.triplet_loss import batch_semi_hard_triplet_loss
from model_demo.slim.nets import vgg
from model_demo.slim.nets import inception_resnet_v2


def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    images = features
    # images = tf.reshape(images, [-1, params.image_size, params.image_size, 1])
    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)

    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # Compute the embeddings with the model
        embeddings, _, _ = inception_resnet_v2.inception_resnet_v2(inputs=images,
                                                                   is_training=is_training,
                                                                   num_classes=params.embedding_size,
                                                                   create_aux_logits=False,
                                                                   mid_feature='Mixed_6a')

    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin, squared=params.squared)
    elif params.triplet_strategy == "batch_semi_hard":
        loss, fraction = batch_semi_hard_triplet_loss(labels, embeddings, margin=params.margin, squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin, squared=params.squared)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

        if params.triplet_strategy == "batch_all" or params.triplet_strategy == 'batch_semi_hard':
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all" or params.triplet_strategy == 'batch_semi_hard':
        tf.summary.scalar('fraction_positive_triplets', fraction)

    tf.summary.image('train_image', images, max_outputs=1)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
