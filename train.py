"""Train the model"""

import os
import cv2
import argparse
import tensorflow as tf

from utils.input_fn import train_input_fn
from utils.model_fn import model_fn
from utils.utils import Params
from utils.utils import get_data

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='saved_model', help="Experiment directory containing params.json")
parser.add_argument('--image_dir', default='', help="Directory containing the query and gallery folders")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    model_params_path = os.path.join(args.model_dir, 'params.json')
    label_map_path = os.path.join(args.image_dir, 'label_map.pbtxt')
    train_data_dir = os.path.join(args.image_dir, 'gallery/Nestle')
    assert os.path.isfile(model_params_path), "No json configuration file found at {}".format(model_params_path)
    assert os.path.isfile(label_map_path), "No label map file found at {}".format(label_map_path)
    assert os.path.isdir(train_data_dir), "No folder found at {}".format(train_data_dir)
    params = Params(model_params_path)

    # Config the train param
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps,
                                    save_checkpoints_steps=2*params.save_summary_steps,
                                    keep_checkpoint_every_n_hours=0.5)
    # Define the model
    tf.logging.info("Creating the model...")
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    # Process input data
    data_tuple = get_data(train_data_dir, label_map_path)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: train_input_fn(data_tuple, params))
