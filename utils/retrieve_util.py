#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import shutil
import numpy as np
import tensorflow as tf
from utils import scda_utils


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


def preprocess(image_path):
    """
    read image, resize to input_shape or size below 700, zero-means
    :param image_path: image absolute path
    :return:
    """
    mean = [123.68, 116.779, 103.939]  # np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])

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
    #     img = tf.image.resize(img, size=(t_h, t_w))

    img = (img - mean)  # / std  vgg paper only need means don't need std
    batch_img = tf.expand_dims(img, 0)  # batch_size
    # flip_img = tf.image.flip_left_right(batch_img)
    # batch_img = tf.concat((batch_img, flip_img), axis=0)
    return batch_img


def build_gallery(sess, input_node, output_node, image_paths, gallery_data_dir):
    """
    将gallery图片进行特征编码并保存相关数据
    :param sess: a tf.Session() 用来启动模型
    :param input_node: 模型的输入节点，placeholder，用来传入图片
    :param output_node: 模型的输出节点，得到最终结果
    :param image_paths: list,所有图片路径
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
        feature = scda_utils.scda_plus(batch_embedding)
        # feature = scda_utils.scda_flip(batch_embedding)
        # feature = scda_utils.scda_flip_plus(batch_embedding)
        # print(feature.shape)
        feature /= np.linalg.norm(feature, keepdims=True)
        feature_list.append(feature)

    # save feature
    feature_list = np.array(feature_list)
    np.save(os.path.join(gallery_data_dir, 'gallery_features.npy'), feature_list)

    print('Finish building gallery!')
    return feature_list


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
