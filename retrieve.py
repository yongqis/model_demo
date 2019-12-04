#!/usr/bin/env python
# -*- coding:utf-8 -*-\
import os
import argparse
import numpy as np
import tensorflow as tf

from slim.nets import vgg
from utils.retrieve_utils import build_gallery, query
from utils import scda_utils
from utils import data_utils

# slim = tf.contrib.slim

DIR_list = [
    '/home/hnu/workspace/syq/CUB_200_2011',
    '/home/hnu/workspace/syq/Stanford_Dog',
    '/home/hnu/workspace/syq/Oxford_Flower',
    '/home/hnu/workspace/syq/Oxford_Pet',
    '/home/hnu/workspace/syq/Aircraft'
]

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default=DIR_list[4], help="Directory containing image folders")
parser.add_argument('--gallery_dir', default='saved_gallery', help='')
parser.add_argument('--model_path', default=r'saved_model/vgg_16.ckpt', help='')

args = parser.parse_args()


def retrieve(image_dir, gallery_dir, model_path, gallery_encode, feature_code):
    """
    :param image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_dir: build_gallery将数据保存在此目录，single_query从此目录读取数据
    :param model_path: 加载指定模型
    :param gallery_encode: True 则进行特征编码，否则加载已经保存的文件
    :param feature_code: 1-scda ,2-scda_flip,3-scda_plus,4-scda_flip_plus.input_batch and output_layer also different
    :return:
    """
    # check dir
    assert os.path.isdir(gallery_dir), 'no directory name {}'.format(gallery_dir)  # 保存gallery的文件夹
    assert os.path.isdir(image_dir), 'no directory name {}'.format(image_dir)  # 数据集文件夹
    assert os.path.isfile(model_path), 'model path not given!'

    # build model
    input_shape = (None, None, None, 3)
    images = tf.placeholder(shape=input_shape, dtype=tf.float32)
    final_output, feature_dict = vgg.vgg_16(
        inputs=images,
        num_classes=None,
        is_training=False)
    # print(feature_dict)
    feature_1 = feature_dict['vgg_16/pool5']
    feature_2 = feature_dict['vgg_16/conv5/conv5_2']
    # final output node depend on feature code
    if feature_code == 1 or feature_code == 2:
        feature = feature_1
    else:
        feature = [feature_1, feature_2]

    # restore 过滤掉一些不需要加载参数 返回dict可以将保存的变量对应到模型中新的变量，返回list直接加载
    include_vars_map = None
    saver = tf.train.Saver(include_vars_map)

    # define session
    with tf.Session() as sess:
        # load param
        sess.run(tf.global_variables_initializer())
        print(model_path)
        saver.restore(sess, model_path)

        # data_set
        query_im_paths, query_labels, gallery_im_paths, gallery_labels = data_utils.split_dataset(image_dir)

        # gallery特征提取或加载
        if gallery_encode:
            gallery_features = build_gallery(sess, images, feature, feature_code, gallery_im_paths, gallery_dir)
        else:
            gallery_features = np.load(os.path.join(gallery_dir, 'gallery_features.npy'))

        # 开始检索
        query(sess, images, feature, feature_code, query_im_paths, gallery_features, query_labels, gallery_labels)



if __name__ == '__main__':
    retrieve(image_dir=args.image_dir,
             gallery_dir=args.gallery_dir,
             model_path=args.model_path,
             gallery_encode=True,
             feature_code=2)
