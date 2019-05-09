#!/usr/bin/env python
# -*- coding:utf-8 -*-\
import os
import cv2
import numpy as np
import tensorflow as tf
from slim.nets import vgg
from slim.nets import inception_resnet_v2
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from utils.utils import Params
from utils.utils import get_ab_path, get_dict, compute_topk


data_dir_dict = {'gallery': r'D:\Picture\Nestle\Nestle_for_retrieval\train',
                 'query': r'D:\Picture\Nestle\Nestle_for_retrieval\query'}


def preprocess(image_path, input_shape):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_shape[1], input_shape[2]))  # resize
    img = img.astype(np.float32)  # keras for_mat
    img = (2.0 / 255.0) * img - 1.0
    batch_img = np.expand_dims(img, 0)  # batch_size
    return batch_img


def build_gallery(sess, input_shape, intput, output, data_dir):
    print('Start building gallery...')

    if not os.path.isdir(data_dir):
        print('dir cannot find')
    images_dir = data_dir_dict['gallery']
    truth_image_dict = get_dict(images_dir)  # 将同一类别的所有图片的路径存为字典
    image_paths = get_ab_path(images_dir)  # 文件目录下所有图片的绝对路径

    feature_list = []
    for i, image_path in enumerate(image_paths):
        print(i+1)
        batch_img = preprocess(image_path, input_shape)
        batch_embedding = sess.run(output, feed_dict={intput: batch_img})
        embedding = np.squeeze(batch_embedding)  # 去掉batch维
        feature_list.append(embedding)  # 加入list

    # save feature
    feature_list = np.array(feature_list)
    joblib.dump(truth_image_dict, os.path.join(data_dir, 'label_dict.pkl'))
    joblib.dump(feature_list, os.path.join(data_dir, 'gallery_features.pkl'))
    joblib.dump(image_paths, os.path.join(data_dir, 'gallery_imagePaths.pkl'))

    print('Finish building gallery!')


def singel_query(sess, top_k, input_shape, input, output, data_dir):
    image_dir = data_dir_dict['query']
    query_image_paths = get_ab_path(image_dir)  # 文件目录下所有图片的绝对路径
    # load gallery
    lablel_map = joblib.load(os.path.join(data_dir, 'label_dict.pkl'))
    gallery_features = joblib.load(os.path.join(data_dir, 'gallery_features.pkl'))
    gallery_image_paths = joblib.load(os.path.join(data_dir, 'gallery_imagePaths.pkl'))

    sum_right = 0
    count = 0
    precision_list = []
    # pass image
    for i, query_image_path in enumerate(query_image_paths):
        print(i + 1)
        # precess image
        batch_img = preprocess(query_image_path, input_shape)
        # get embeddings
        batch_embedding = sess.run(output, feed_dict={input: batch_img})
        embedding = np.squeeze(batch_embedding)  # 去掉batch维
        # retrieve
        query_feature = embedding
        query_label = query_image_path.split('\\')[-2]
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


def retrieve(data_dir, model_dir, top_k, only_query, model_saved_num=None):
    params_path = os.path.join(model_dir, 'params.json')
    params = Params(params_path)
    if model_saved_num:
        model_path = os.path.join(model_dir, 'model.ckpt-'+str(model_saved_num))
    else:
        model_path = tf.train.get_checkpoint_state(model_dir).model_checkpoint_path

    input_shape = (None, params.image_size, params.image_size, 3)

    # build graph
    with tf.variable_scope('model'):
        images = tf.placeholder(dtype=tf.float32, shape=input_shape)
        embeddings, _, _ = inception_resnet_v2.inception_resnet_v2(inputs=images,
                                                                   is_training=False,
                                                                   num_classes=params.embedding_size,
                                                                   create_aux_logits=False,
                                                                   mid_feature=params.final_endpoint)

    with tf.Session() as sess:
        # load param
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        if not only_query:
            build_gallery(sess, input_shape, images, embeddings, data_dir)
        singel_query(sess, top_k, input_shape, images, embeddings, data_dir)


if __name__ == '__main__':
    retrieve(data_dir='',
             model_dir='',
             top_k=1,
             only_query=False,
             model_saved_num=None)

