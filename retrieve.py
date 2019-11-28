#!/usr/bin/env python
# -*- coding:utf-8 -*-\
import os
import argparse
import numpy as np
import tensorflow as tf

from slim.nets import vgg
from utils import retrieve_util
from utils import scda_utils

slim = tf.contrib.slim
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='saved_model', help="Experiment directory containing params.json and ckpt")
parser.add_argument('--image_dir', default=r'D:\model_data\raw\CUB_200_2011',
                    help="Directory containing the query image and gallery image folders")
parser.add_argument('--data_dir', default='saved_data', help='')
parser.add_argument('--load_model_path', default=r'saved_model/vgg_16.ckpt', help='')

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

    # build model
    # input_shape = (None, None, None, 3)
    im_path = tf.placeholder(dtype=tf.string)
    images = retrieve_util.preprocess(im_path)
    final_output, feature_dict = vgg.vgg_16(
        inputs=images,
        num_classes=None,
        is_training=False)
    #  CNN output encode & normalize
    # print(feature_dict)
    feature_1 = feature_dict['vgg_16/pool5']
    feature_2 = feature_dict['vgg_16/conv5/conv5_2']
    feature = feature_1

    # restore 过滤掉一些不需要加载参数 返回dict可以将保存的变量对应到模型中新的变量，返回list直接加载
    include_vars_map = None
    saver = tf.train.Saver(include_vars_map)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(saved_model_path)
        saver.restore(sess, saved_model_path)
        query_image_paths, query_labels, gallery_image_paths, gallery_labels = retrieve_util.split_data(base_image_dir)
        # gallery特征提取或加载
        if gallery_encode:
            gallery_features = retrieve_util.build_gallery(sess, im_path, feature, gallery_image_paths, gallery_data_dir)
        else:
            gallery_features = np.load(os.path.join(gallery_data_dir, 'gallery_features.npy'))

        # 开始检索
        query_num = len(query_image_paths)
        query_labels = np.array(query_labels)
        gallery_labels = np.array(gallery_labels)

        top_1 = 0.0
        top_5 = 0.0
        feature_list = []
        for i, query_image_path in enumerate(query_image_paths):
            print('---------')
            print('{}/{}'.format(i, query_num))
            # get feature map
            batch_embedding = sess.run(feature, feed_dict={im_path: query_image_path})
            # scda encode
            query_feature = scda_utils.scda(batch_embedding)
            # query_feature = scda_utils.scda_plus(batch_embedding)
            # query_feature = scda_utils.scda_flip(batch_embedding)
            # query_feature = scda_utils.scda_flip_plus(batch_embedding)
            query_feature /= np.linalg.norm(query_feature, keepdims=True)
            # 计算相似度，并排序
            cos_sim = np.dot(query_feature, gallery_features.T)
            cos_sim = 0.5 + 0.5 * cos_sim  # 归一化， [-1, 1] --> [0, 1]
            sorted_indices = np.argsort(-cos_sim)  # 值越大相似度越大，因此添加‘-’升序排序
            # 统计检索结果AP top1 top5
            query_label = query_labels[i]
            k_gallery_label = gallery_labels[sorted_indices[:5]]
            # 计算top1的AP
            if query_label == k_gallery_label[0]:
                top_1 += 1
            # 计算top5的AP
            correct=0
            ap=0
            for i in range(5):
                if query_label == k_gallery_label[i]:
                    correct+=1
                ap+=(correct/(i+1))
            top_5+=(ap/5)
            print("top1-AP:%f | top5-AP: %f" %(top_1, ap/5))
            # feature_list.append(embedding)
        # feature_list = np.array(feature_list)
        # 统计mAP
        print('top1-mAP:', round(top_1 / query_num, 5))
        print('top5-mAP:', round(top_5 / query_num, 5))


if __name__ == '__main__':
    retrieve(model_dir=args.model_dir,
             base_image_dir=args.image_dir,
             gallery_data_dir=args.data_dir,
             saved_model_path=args.load_model_path,
             gallery_encode=True,
             )
