"""Train the model"""

import argparse
import os

import tensorflow as tf

from utils.input_fn import train_input_fn
from utils.input_fn import test_input_fn
from utils.model_fn import model_fn
from utils.utils import Params
from utils import label_map_util

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='saved_model', help="Experiment directory containing params.json")
parser.add_argument('--label_map_path', default=r'D:\Pycharm\joint_retrieval\data\all_class\label_map.pbtxt')
parser.add_argument('--train_data_dir', default='D:\\Picture\\Nestle\\Nestle_for_retrieval\\train',
                    help="Directory containing the dataset")
# parser.add_argument('--test_data_dir')


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    label_map = label_map_util.get_label_map_dict(args.label_map_path)
    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=500,
                                    save_checkpoints_steps=1000,
                                    log_step_count_steps=100)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    image_path_list = []
    image_label_list = []
    for cur_folder, sub_folders, sub_files in os.walk(args.train_data_dir):
        for file in sub_files:
            if file.endswith('jpg'):
                image_path_list.append(os.path.join(cur_folder, file))
                image_label_list.append(label_map[os.path.split(cur_folder)[-1]])

    image_list = (image_path_list, image_label_list)
    estimator.train(lambda: train_input_fn(image_list, params))

    # # Evaluate the model on the test set
    # tf.logging.info("Evaluation on test set.")
    # res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    # for key in res:
    #     print("{}: {}".format(key, res[key]))
