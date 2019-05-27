#!/usr/bin/env python
# -*- coding:utf-8 -*-\
import os
import argparse
import tensorflow as tf
from slim.nets import vgg
from utils.utils import Params
from utils import retrieve_util

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='saved_model', help="Experiment directory containing params.json and ckpt")
parser.add_argument('--image_dir', default='', help="Directory containing the query image and gallery image folders")
parser.add_argument('--data_dir', default='saved_data', help='')
parser.add_argument('--load_model_path', default=None, help='')


def retrieve(model_dir, base_image_dir, gallery_data_dir, saved_model_path=None):
    """
    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: build_gallery将数据保存在此目录，single_query从此目录读取数据
    :param model_dir: 目录下保存训练模型的和模型参数文件params.json
    :param saved_model_path: 加载指定模型，如果为None 加载最新模型
    :return:
    """
    # check dir
    assert os.path.isdir(gallery_data_dir), 'no directory name {}'.format(gallery_data_dir)
    assert os.path.isdir(base_image_dir), 'no directory name {}'.format(base_image_dir)
    assert os.path.isdir(model_dir), 'no directory name {}'.format(model_dir)
    # init data
    params_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(params_path), 'no params file'
    params = Params(params_path)
    # 默认加载最新训练的模型
    model_path = tf.train.get_checkpoint_state(model_dir).model_checkpoint_path
    # 或者加载指定模型
    if saved_model_path:
        model_path = saved_model_path
    #
    input_shape = (None, params.image_size, params.image_size, 3)

    # build graph
    with tf.variable_scope('model'):
        images = tf.placeholder(dtype=tf.float32, shape=input_shape)
        final_output, feature_dict = vgg.vgg_16(
            inputs=images,
            num_classes=256,
            is_training=False,
            dropout_keep_prob=0.7,
            spatial_squeeze=True,
            scope='vgg_16',
            fc_conv_padding='VALID',
            global_pool=False)

    # encode & normalize
    embeddings = retrieve_util.encode(feature_dict['layer_name'])  # or

    # run
    with tf.Session() as sess:
        # load model
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        #
        if not params.only_query:
            retrieve_util.build_gallery(sess, input_shape, images, embeddings, base_image_dir, gallery_data_dir)
        retrieve_util.image_query(sess, input_shape, images, embeddings, base_image_dir, gallery_data_dir, params.top_k)


if __name__ == '__main__':
    args = parser.parse_args()
    retrieve(model_dir=args.model_dir,
             base_image_dir=args.image_dir,
             gallery_data_dir=args.data_dir,
             saved_model_path=args.load_model_path)

