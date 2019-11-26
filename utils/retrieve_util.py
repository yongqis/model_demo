#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
from utils import scda_utils
from sklearn.externals import joblib
from utils.config import get_ab_path, get_dict


def split_data(datafile_root):
    # 划分数据集
    img_txt_file = open(os.path.join(datafile_root, 'images.txt'))
    label_txt_file = open(os.path.join(datafile_root, 'image_class_labels.txt'))
    train_val_file = open(os.path.join(datafile_root, 'train_test_split.txt'))
    img_name_list = []
    for line in img_txt_file:
        img_name_list.append(line[:-1].split(' ')[-1])
    label_list = []
    for line in label_txt_file:
        label_list.append(int(line[:-1].split(' ')[-1]) - 1)
    train_test_list = []
    for line in train_val_file:
        train_test_list.append(int(line[:-1].split(' ')[-1]))
    # 按照文件中路径划分为train test数据集
    train_image_paths = [os.path.join(datafile_root, 'images', x) for i, x in zip(train_test_list, img_name_list)
                         if i]
    test_image_paths = [os.path.join(datafile_root, 'images', x) for i, x in zip(train_test_list, img_name_list) if
                        not i]

    train_image_labels = [x for i, x in zip(train_test_list, label_list) if i]
    test_image_labels = [x for i, x in zip(train_test_list, label_list) if not i]
    return test_image_paths, test_image_labels, train_image_paths, train_image_labels


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def preprocess(image_path, input_shape=None):
    """
    read image, resize to input_shape, zero-means
    :param image_path:
    :param input_shape:
    :return:
    """
    mean = [123.68, 116.779, 103.939] # np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # tf处理图片
    img_raw_data = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_raw_data)
    img = tf.to_float(img)
    shape = tf.shape(img)
    if shape[-1] == 1:
        print("grayscale image will be convert to rgb")
        img = tf.image.grayscale_to_rgb(img)
    # min_side = tf.minimum(shape[0], shape[1])
    # if min_side > 700:
    #     t_h = shape[0] * 700 // min_side
    #     t_w = shape[1] * 700 // min_side
    img = (img - mean) # / std
    # img = _mean_image_subtraction(img, mean)
    batch_img = tf.expand_dims(img, 0)  # batch_size
    return batch_img


def encode(feature):
    """

    :param feature: 4-D tensor [batch_size, height, width, channel]
    :return: 2-D tensor [batch_size, channel]
    """

    feature1 = tf.reduce_mean(feature, axis=[1, 2])
    feature2 = tf.reduce_max(feature, axis=[1, 2])

    feature = tf.squeeze(tf.concat([feature1, feature2], axis=-1))
    embeddings = tf.nn.l2_normalize(feature, axis=0)
    return embeddings


def build_gallery(sess, input_shape, input_node, output_node, image_paths, gallery_data_dir):
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

    nums = len(image_paths)
    feature_list = []
    for i, image_path in enumerate(image_paths):
        print('{}/{}'.format(i + 1, nums))
        batch_embedding = sess.run(output_node, feed_dict={input_node: image_path})
        embedding = np.squeeze(batch_embedding)  # 去掉batch维
        query_feature, _ = scda_utils.scda(embedding)
        feature_list.append(query_feature)

    # save feature
    feature_list = np.array(feature_list)
    np.save(os.path.join(gallery_data_dir, 'gallery_features.npy'), feature_list)

    print('Finish building gallery!')
    return feature_list


def image_query(sess, input_shape, input_node, output_node, base_image_dir, gallery_data_dir, top_k=5,
                sim_threshold=0.5):
    """

    :param sess: a tf.Session() 用来启动模型
    :param top_k: 检索结果取top-k个 计算准确率Acc = (TN + TP)/(N + P)
    :param input_shape: 图片resize的目标大小，和模型的placeholder保持一致
    :param input_node: 模型的输入节点，placeholder，用来传入图片
    :param output_node: 模型的输出节点，得到最终结果
    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: ；用来读取build_gallery保存的数据
    :param sim_threshold : 相似度阈值
    :return:
    """
    query_image_dir = os.path.join(base_image_dir, 'query')
    query_image_paths = get_ab_path(query_image_dir)  # 得到文件夹内所有图片的绝对路径
    query_num = len(query_image_paths)
    saved_error_dir = os.path.join(gallery_data_dir, 'error_image')  # 该文件夹 保存检索错误的图片
    if not os.path.isdir(saved_error_dir):
        saved_error_dir = None
    # load gallery
    lablel_map = joblib.load(os.path.join(gallery_data_dir, 'label_dict.pkl'))
    gallery_features = joblib.load(os.path.join(gallery_data_dir, 'gallery_features.pkl'))
    gallery_image_paths = joblib.load(os.path.join(gallery_data_dir, 'gallery_imagePaths.pkl'))
    # statistics params
    sum_list = []
    for i, query_image_path in enumerate(query_image_paths):
        # if i == 100:
        #     break
        print('---------')
        print('{}/{}'.format(i, query_num))
        # precess image
        batch_img = preprocess(query_image_path, input_shape)
        # get embedding image
        batch_embedding = sess.run(output_node, feed_dict={input_node: batch_img})
        embedding = np.squeeze(batch_embedding)  # 去掉batch维
        # 计算余弦相似度，归一化，并排序
        query_feature = embedding
        cos_sim = np.dot(query_feature, gallery_features.T)
        cos_sim = 0.5 + 0.5 * cos_sim
        sorted_indices = np.argsort(-cos_sim)
        # 开始检查检索结果
        query_label = os.path.split(os.path.dirname(query_image_path))[-1]  # 查询图片的真实类别
        truth_image_paths = lablel_map[query_label]  # 与查询图片同类别的所有图片路径，即检索正确时，结果应该在此范围内

        saved_error_label_dir = None  # 将检索错误的图片保存在该类别文件夹内
        if saved_error_dir:
            saved_error_label_dir = os.path.join(saved_error_dir, query_label)
            if not os.path.isdir(saved_error_label_dir):
                os.makedirs(saved_error_label_dir)
        res_list = get_topk(sorted_indices, gallery_image_paths, truth_image_paths, query_image_path,
                            top_k, saved_error_dir=saved_error_label_dir, query_id=i)
        sum_list.append(res_list)

    sum_arr = np.array(sum_list)
    ss = np.sum(sum_arr, axis=0)
    ss = ss / sum_arr.shape[0]

    for i, value in enumerate(ss):
        print('top-{} acc:{}'.format(i + 1, value))


def get_topk(score, gallery_images, truth_images, query_image, top_k=5, saved_error_dir=None, query_id=None):
    """
    根据相似度得分，从高到低依次检查检索结果是否正确，
    并将结果保存的res_dict中，key为label名，
    最后使用多数表决规则决定最终的检索类别
    :param score: 排序后的相似度得分-值为对应的索引
    :param gallery_images: 数据集中所有图片路径
    :param truth_images: 正确范围的图片路径
    :param query_image: 查询的图片路径
    :param top_k:
    :param saved_error_dir:
    :param query_id:
    :return:
    """

    res_dict = {}
    stage_list = []

    bias = 0  # 如果查询到自身图片，需要跳过，
    for i, index in enumerate(score):
        i += bias
        if i == top_k:
            break
        res_image = gallery_images[index]  # 检索出来的图片

        # 查找正确
        if res_image in truth_images:
            # 文件名不同，不是同一张图片，则结果正确
            if os.path.split(res_image)[-1] != os.path.split(query_image)[-1]:
                res_dict.setdefault('right_label', 0)
                res_dict['right_label'] += 1
            # 文件名相同，找到自己，忽略，查看下一个
            else:
                bias = -1
                # print('现在是top-{}，检索到了自己'.format(i))
                # if i != 0:
                # print(query_image)
                continue
        # 查找错误，拷贝出来图片进行分析
        else:

            truth_label = os.path.split(os.path.dirname(query_image))[-1]
            error_label = os.path.split(os.path.dirname(res_image))[-1]
            # print('现在是top-{}，检索错误'.format(i))
            res_dict.setdefault(error_label, 0)
            res_dict[error_label] += 1
            if saved_error_dir:
                # 查询图片处理
                copy_path = os.path.join(saved_error_dir, os.path.basename(query_image))
                new_name = os.path.join(saved_error_dir, str(query_id) + '_' + str(0) + '_' + truth_label
                                        + '_' + os.path.basename(query_image))
                if not os.path.isfile(new_name):
                    # 1.复制
                    shutil.copy(query_image, copy_path)
                    # 2.改名
                    os.rename(copy_path, new_name)
                # 错误图片处理
                copy_path = os.path.join(saved_error_dir, os.path.basename(res_image))
                new_name = os.path.join(saved_error_dir, str(query_id) + '_' + str(i + 1) + '_' + error_label
                                        + '_' + os.path.basename(res_image))
                if not os.path.isfile(new_name):
                    # 1.复制
                    shutil.copy(res_image, copy_path)
                    # 2.改名
                    os.rename(copy_path, new_name)

        # 检查当前top-i轮的多数项作为结果是否正负，stage_list.append(1 or 0)
        max_times = 0
        max_label = ''
        for key, value in res_dict.items():
            if value > max_times:
                max_label = key
        max_label = 'right_label' if max_times is 1 else max_label
        stage_list.append(int(max_label == 'right_label'))
    return stage_list


def image_query_new(sess, input_shape, input_node, output_node, query_image_paths, query_label, gallery_label,
                    gallery_features, top_k=5):
    """

    :param sess: a tf.Session() 用来启动模型
    :param top_k: 检索结果取top-k个 计算准确率Acc = (TN + TP)/(N + P)
    :param input_shape: 图片resize的目标大小，和模型的placeholder保持一致
    :param input_node: 模型的输入节点，placeholder，用来传入图片
    :param output_node: 模型的输出节点，得到最终结果
    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: ；用来读取build_gallery保存的数据
    :param sim_threshold : 相似度阈值
    :return:
    """
    query_num = len(query_image_paths)
    query_label = np.array(query_label)
    gallery_label = np.array(gallery_label)
    # statistics params
    top_1 = 0
    top_5 = 0
    for i, query_image_path in enumerate(query_image_paths):
        # if i == 100:
        #     break
        print('---------')
        print('{}/{}'.format(i, query_num))
        # precess image
        batch_img = preprocess(query_image_path, input_shape)
        # get embedding image
        batch_embedding = sess.run(output_node, feed_dict={input_node: batch_img})
        embedding = np.squeeze(batch_embedding)  # 去掉batch维
        # 计算余弦相似度，归一化，并排序
        query_feature = embedding
        cos_sim = np.dot(query_feature, gallery_features.T)
        cos_sim = 0.5 + 0.5 * cos_sim
        sorted_indices = np.argsort(-cos_sim)
        # 检查 检索结果

        k_gallery_label = gallery_label[sorted_indices[:top_k]]
        if query_label[i] == k_gallery_label[0]:
            top_1 += 1
            top_5 += 1
            print("all true")
        elif query_label[i] in k_gallery_label:
            top_5 += 1
    print(top_1 / query_num)
    print(top_5 / query_num)


def similiar(feature, query_label, gallery_features, gallery_label, top1, top5):
    # 计算余弦相似度，归一化，并排序
    query_feature = feature
    cos_sim = np.dot(query_feature, gallery_features.T)
    cos_sim = 0.5 + 0.5 * cos_sim
    sorted_indices = np.argsort(-cos_sim)
    # 检查 检索结果

    k_gallery_label = gallery_label[sorted_indices[:5]]
    if query_label == k_gallery_label[0]:
        top1 += 1
        top5 += 1
        print("all true")
    elif query_label in k_gallery_label:
        top5 += 1
    return top1, top5
