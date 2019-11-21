#!/usr/bin/env python
# -*- coding:utf-8 -*-\
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

from slim.nets import vgg
from utils.config import Params
from utils import retrieve_util
from utils import scda_utils

slim = tf.contrib.slim
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='saved_model', help="Experiment directory containing params.json and ckpt")
parser.add_argument('--image_dir', default=r'D:\model_data\raw\CUB_200_2011', help="Directory containing the query image and gallery image folders")
parser.add_argument('--data_dir', default='saved_data', help='')
parser.add_argument('--load_model_path', default=r'saved_model\vgg_16.ckpt', help='')

args = parser.parse_args()

def retrieve(model_dir, base_image_dir, gallery_data_dir, gallery_encode, saved_model_path=None):
    """
    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: build_gallery将数据保存在此目录，single_query从此目录读取数据
    :param model_dir: 目录下保存训练模型的和模型参数文件params.json
    :param saved_model_path: 加载指定模型，如果为None 加载最新模型
    :return:
    """
    # check dir
    assert os.path.isdir(gallery_data_dir), 'no directory name {}'.format(gallery_data_dir)  # 保存gallery的文件夹
    assert os.path.isdir(base_image_dir), 'no directory name {}'.format(base_image_dir)  # 数据集文件夹
    assert os.path.isdir(model_dir), 'no directory name {}'.format(model_dir)  # 模型参数文件夹
    params_path = os.path.join(model_dir, 'params.json')  # 模型参数文件
    assert os.path.isfile(params_path), 'no params file'
    # 初始化参数对象
    params = Params(params_path)

    # build model
    input_shape = (None, None, None, 3)
    images = tf.placeholder(dtype=tf.float32, shape=input_shape)
    final_output, feature_dict = vgg.vgg_16(
        inputs=images,
        num_classes=None,
        is_training=False)
    #  CNN output encode & normalize
    feature = feature_dict['vgg_16/pool5']
    embeddings = retrieve_util.encode(feature)

    # restore 默认加载目录下最新训练的模型 或者加载指定模型
    # model_path = tf.train.get_checkpoint_state(model_dir).model_checkpoint_path
    # if saved_model_path:
    #   model_path = saved_model_path

    # restore 过滤掉一些不需要加载参数 返回dict可以将保存的变量对应到模型中新的变量，返回list直接加载
    include_vars_map = None
    saver = tf.train.Saver(include_vars_map)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(saved_model_path)
        saver.restore(sess, saved_model_path)
        query_image_paths, query_labels, gallery_image_paths, gallery_labels = retrieve_util.split_data(base_image_dir)
        # gallery特征提取
        if gallery_encode:
             gallery_features = retrieve_util.build_gallery(sess, input_shape, images, embeddings, gallery_image_paths, gallery_data_dir)
        else:
            gallery_features = joblib.load(os.path.join(gallery_data_dir, 'gallery_features.pkl'))
        # 开始检索
        # retrieve_util.image_query_new(sess, input_shape, images, embeddings, query_image_paths, query_labels, gallery_labels, gallery_features, params.top_k)

        query_num = len(query_image_paths)
        query_label = np.array(query_labels)
        gallery_label = np.array(gallery_labels)
        # statistics params
        top_1 = 0
        top_5 = 0
        for i, query_image_path in enumerate(query_image_paths):
            # if i == 100:
            #     break
            print('---------')
            print('{}/{}'.format(i, query_num))
            # precess image
            batch_img = retrieve_util.preprocess(query_image_path, input_shape)
            # get embedding image
            batch_embedding = sess.run(embedding, feed_dict={images: batch_img})

            # 计算余弦相似度，归一化，并排序
            embedding = np.squeeze(batch_embedding)  # 去掉batch维
            query_feature = embedding
            cos_sim = np.dot(query_feature, gallery_features.T)
            cos_sim = 0.5 + 0.5 * cos_sim
            sorted_indices = np.argsort(-cos_sim)
            # 检查 检索结果

            k_gallery_label = gallery_label[sorted_indices[:params.top_k]]
            if query_label[i] == k_gallery_label[0]:
                top_1 += 1
                top_5 += 1
                print("all true")
            elif query_label[i] in k_gallery_label:
                top_5 += 1
        print(top_1 / query_num)
        print(top_5 / query_num)

if __name__ == '__main__':
    retrieve(model_dir=args.model_dir,
             base_image_dir=args.image_dir,
             gallery_data_dir=args.data_dir,
             saved_model_path=args.load_model_path,
             gallery_encode=True,
    )

