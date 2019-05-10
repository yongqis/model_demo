#!/usr/bin/env python
# -*- coding:utf-8 -*-\
import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from slim.nets import vgg
from slim.nets import inception_resnet_v2
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from utils.utils import Params
from utils.utils import get_ab_path, get_dict, compute_topk

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='saved_model', help="Experiment directory containing params.json and ckpt")
parser.add_argument('--image_dir', default='', help="Directory containing the query image and gallery image folders")
parser.add_argument('--data_dir', default='saved_data', help='')
parser.add_argument('--load_model_num', default=None, help='')


def preprocess(image_path, input_shape):
    """
    read image, resize to input_shape, zero-means
    :param image_path:
    :param input_shape:
    :return:
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_shape[1], input_shape[2]))  # resize
    img = img.astype(np.float32)  # keras for_mat
    img = (2.0 / 255.0) * img - 1.0
    batch_img = np.expand_dims(img, 0)  # batch_size
    return batch_img


def build_gallery(sess, input_shape, input_node, output_node, base_image_dir, gallery_data_dir):
    """
    将gallery图片进行特征编码并保存相关数据
    :param sess: a tf.Session() 用来启动模型
    :param input_shape: 图片resize的目标大小，和模型的placeholder保持一致
    :param input_node: 模型的输入节点，placeholder，用来传入图片
    :param output_node: 模型的输出节点，得到最终结果
    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: gallery文件夹内的图片经模型提取的特征、图片路径以及图片路径字典都将保存在目录下
    :return:
    """
    print('Start building gallery...')

    assert os.path.isdir(gallery_data_dir), 'dir: {} cannot find'.format(gallery_data_dir)

    images_dir = os.path.join(base_image_dir, 'gallery')
    assert os.path.isdir(images_dir), 'dir: {} cannot find'.format(images_dir)
    truth_image_dict = get_dict(images_dir)  # 将同一类别的所有图片的路径存为字典
    image_paths = get_ab_path(images_dir)  # 文件目录下所有图片的绝对路径

    feature_list = []
    for i, image_path in enumerate(image_paths):
        print(i+1)
        batch_img = preprocess(image_path, input_shape)
        batch_embedding = sess.run(output_node, feed_dict={input_node: batch_img})
        embedding = np.squeeze(batch_embedding)  # 去掉batch维
        feature_list.append(embedding)  # 加入list

    # save feature
    feature_list = np.array(feature_list)
    joblib.dump(truth_image_dict, os.path.join(gallery_data_dir, 'label_dict.pkl'))
    joblib.dump(feature_list, os.path.join(gallery_data_dir, 'gallery_features.pkl'))
    joblib.dump(image_paths, os.path.join(gallery_data_dir, 'gallery_imagePaths.pkl'))

    print('Finish building gallery!')


def single_query(sess, top_k, input_shape, input_node, output_node, base_image_dir, gallery_data_dir):
    """

    :param sess: a tf.Session() 用来启动模型
    :param top_k: 检索结果取top_k个 计算准确率
    :param input_shape: 图片resize的目标大小，和模型的placeholder保持一致
    :param input_node: 模型的输入节点，placeholder，用来传入图片
    :param output_node: 模型的输出节点，得到最终结果
    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: ；用来读取build_gallery保存的数据
    :return:
    """
    image_dir = os.path.join(base_image_dir, 'query')
    query_image_paths = get_ab_path(image_dir)  # 文件目录下所有图片的绝对路径
    # load gallery
    lablel_map = joblib.load(os.path.join(gallery_data_dir, 'label_dict.pkl'))
    gallery_features = joblib.load(os.path.join(gallery_data_dir, 'gallery_features.pkl'))
    gallery_image_paths = joblib.load(os.path.join(gallery_data_dir, 'gallery_imagePaths.pkl'))

    sum_right = 0
    count = 0
    precision_list = []
    # pass image
    for i, query_image_path in enumerate(query_image_paths):
        print(i + 1)
        # precess image
        batch_img = preprocess(query_image_path, input_shape)
        # get embeddings
        batch_embedding = sess.run(output_node, feed_dict={input_node: batch_img})
        embedding = np.squeeze(batch_embedding)  # 去掉batch维
        # retrieve
        query_feature = embedding
        query_label = os.path.split(os.path.dirname(query_image_path))[-1]
        cos_sim = np.dot(query_feature, gallery_features.T)
        cos_sim = 0.5 + 0.5 * cos_sim
        sorted_indices = np.argsort(-cos_sim)

        truth_image_paths = lablel_map[query_label]
        is_right = compute_topk(sorted_indices, gallery_image_paths, truth_image_paths, query_image_path, top_k)

        sum_right += is_right
        count += 1
        precision_list.append(sum_right / count)

    precision = sum_right / count
    print("TOP-k:", precision)


def retrieve(model_dir, base_image_dir, gallery_data_dir, model_saved_num=None):
    """

    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: build_gallery将数据保存在此目录，single_query从此目录读取数据
    :param model_dir: 目录下保存训练模型的和模型参数文件params.json
    :param model_saved_num: 加载指定训练次数的模型，如果为None 加载最新模型
    :return:
    """
    # check dir

    # init data
    params_path = os.path.join(model_dir, 'params.json')
    params = Params(params_path)
    if model_saved_num:
        model_path = os.path.join(model_dir, 'model.ckpt-' + model_saved_num)
    else:
        model_path = tf.train.get_checkpoint_state(model_dir).model_checkpoint_path

    input_shape = (None, params.image_size, params.image_size, 3)
    top_k = params.top_k
    only_query = params.only_query
    # build graph
    with tf.variable_scope('model'):
        images = tf.placeholder(dtype=tf.float32, shape=input_shape)
        embeddings, _, _ = inception_resnet_v2.inception_resnet_v2(inputs=images,
                                                                   is_training=False,
                                                                   num_classes=params.embedding_size,
                                                                   create_aux_logits=False,
                                                                   base_final_endpoint=params.final_endpoint)

    with tf.Session() as sess:
        # load param
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        if not only_query:
            build_gallery(sess, input_shape, images, embeddings, base_image_dir, gallery_data_dir)
        single_query(sess, top_k, input_shape, images, embeddings, base_image_dir, gallery_data_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    retrieve(model_dir=args.model_dir,
             base_image_dir=args.image_dir,
             gallery_data_dir=args.data_dir,
             model_saved_num=args.load_model_num)

