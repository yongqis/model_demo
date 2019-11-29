#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import tensorflow_datasets as tfds

def split_dataset(root_dir, data_set):
    """
    :param root_dir:
    :param data_set: 数据集名称， CUB, DOG, FLOWER, AIR,
    :return:
    """
    assert data_set is not None, 'param data_set can not None'

    if data_set is "CUB":
        return cub_split(root_dir)
    elif data_set is "DOG":
        return dog_split(root_dir)
    else:
        print("No such data set!!!")
        return None


def cub_split(root_dir):
    """
    :param datafile_root: data_set root dir, include images dir and three txt files.
    """
    # 划分数据集
    img_txt_file = open(os.path.join(root_dir, 'images.txt'))
    label_txt_file = open(os.path.join(root_dir, 'image_class_labels.txt'))
    train_val_file = open(os.path.join(root_dir, 'train_test_split.txt'))
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
    train_image_paths = [os.path.join(root_dir, 'images', x) for i, x in zip(train_test_list, img_name_list)
                         if i]
    test_image_paths = [os.path.join(root_dir, 'images', x) for i, x in zip(train_test_list, img_name_list) if
                        not i]

    train_image_labels = [x for i, x in zip(train_test_list, label_list) if i]
    test_image_labels = [x for i, x in zip(train_test_list, label_list) if not i]
    return test_image_paths, test_image_labels, train_image_paths, train_image_labels


def dog_split(root_dir):

    # get train file list
    train_list_file = os.path.join(root_dir, 'train_list.mat')
    with tf.io.gfile.GFile(train_list_file, "rb") as f:
        parsed_mat_arr = tfds.core.lazy_imports.scipy.io.loadmat(f, squeeze_me=True)
        train_image_paths = [os.path.join(root_dir, 'images/'+im_path) for im_path in parsed_mat_arr['file_list']]
        train_image_labels = parsed_mat_arr['labels']

    # get test file list
    test_list_file = os.path.join(root_dir, 'test_list.mat')
    with tf.io.gfile.GFile(test_list_file, "rb") as f:
        parsed_mat_arr = tfds.core.lazy_imports.scipy.io.loadmat(f, squeeze_me=True)
        test_image_paths = [os.path.join(root_dir, 'images/'+im_path) for im_path in parsed_mat_arr['file_list']]
        test_image_labels = parsed_mat_arr['labels']

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
    flip_img = tf.image.flip_left_right(batch_img)
    batch_img = tf.concat((batch_img, flip_img), axis=0)
    return batch_img
