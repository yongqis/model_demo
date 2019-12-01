#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def split_dataset(root_dir):
    """
    :param root_dir:
    :param data_set: 数据集名称， CUB, DOG, FLOWER, AIR,
    :return:
    """
    # assert data_set is not None, 'param data_set can not None'
    data_set = os.path.basename(root_dir)
    if "CUB" in data_set:
        return cub_split(root_dir)
    elif "Dog" in data_set:
        return dog_split(root_dir)
    elif "Flower" in data_set:
        return flower_split(root_dir)
    elif "Pet" in data_set:
        return pet_split(root_dir)
    elif "Aircraft" in data_set:
        return air_split(root_dir)
    else:
        print("No such data set!!!, must in 'CUB','DOG','FLO','PET','AIR'.")
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


def flower_split(root_dir):

    label_list_file = os.path.join(root_dir, 'imagelabels.mat')
    with tf.io.gfile.GFile(label_list_file, "rb") as f:
        parsed_mat_arr = tfds.core.lazy_imports.scipy.io.loadmat(f, squeeze_me=True)
        label_arr = parsed_mat_arr['labels']

    split_list_file = os.path.join(root_dir, 'setid.mat')
    with tf.io.gfile.GFile(split_list_file, "rb") as f:
        parsed_mat_arr = tfds.core.lazy_imports.scipy.io.loadmat(f, squeeze_me=True)

        test_image_id = parsed_mat_arr['tstid']
        test_image_paths = [os.path.join(root_dir,'jpg/'+'image_'+ ('%05d'%i)+'.jpg')  for i in test_image_id]
        test_image_labels = [label_arr[i-1] for i in test_image_id]

        train_image_id = np.concatenate((parsed_mat_arr['trnid'],parsed_mat_arr['valid']))
        train_image_paths = [os.path.join(root_dir,'jpg/'+'image_'+ ('%05d'%i)+'.jpg')  for i in train_image_id]
        train_image_labels = [label_arr[i-1] for i in train_image_id]

    return test_image_paths, test_image_labels, train_image_paths, train_image_labels


def pet_split(root_dir):
    train_list_file = os.path.join(root_dir, 'annotations/'+'trainval.txt')
    test_list_file = os.path.join(root_dir, 'annotations/'+'test.txt')
    tr = open(train_list_file)
    te = open(test_list_file)

    train_image_paths=[]
    train_image_labels=[]
    for info in tr.readlines():
        image_path, image_label = info.split(' ')[:2]
        image_path = os.path.join(root_dir, 'images/'+image_path+'.jpg')
        train_image_paths.append(image_path)
        train_image_labels.append(image_label)

    test_image_paths = []
    test_image_labels = []
    for info in te.readlines():
        image_path, image_label = info.split(' ')[:2]
        image_path = os.path.join(root_dir, 'images/' + image_path + '.jpg')
        test_image_paths.append(image_path)
        test_image_labels.append(image_label)

    return test_image_paths, test_image_labels, train_image_paths, train_image_labels


def air_split(root_dir):
    train_list_file = os.path.join(root_dir, 'fgvc-aircraft-2013b/'+'images_variant_trainval.txt')
    test_list_file = os.path.join(root_dir, 'fgvc-aircraft-2013b/'+'images_variant_test.txt')
    tr = open(train_list_file)
    te = open(test_list_file)

    train_image_paths=[]
    train_image_labels=[]
    for info in tr.readlines():
        image_path, image_label = info.split(' ')[:2]
        image_path = os.path.join(root_dir, 'fgvc-aircraft-2013b/images/'+image_path+'.jpg')
        train_image_paths.append(image_path)
        train_image_labels.append(image_label)

    test_image_paths = []
    test_image_labels = []
    for info in te.readlines():
        image_path, image_label = info.split(' ')[:2]
        image_path = os.path.join(root_dir, 'fgvc-aircraft-2013b/images/' + image_path + '.jpg')
        test_image_paths.append(image_path)
        test_image_labels.append(image_label)

    return test_image_paths, test_image_labels, train_image_paths, train_image_labels


def preprocess(image_path):
    """
    read image, resize to input_shape or size below 700, zero-means
    :param image_path: image absolute path
    :return:
    """
    mean = [123.68, 116.779, 103.939]  # np.array([0.485, 0.456, 0.406]) std = np.array([0.229, 0.224, 0.225])
    # cv2处理图片
    print(image_path)
    img_raw_data = cv2.imread(image_path)

    shape = img_raw_data.shape
    if shape[-1]==1:
        img = cv2.cvtColor(img_raw_data, cv2.COLOR_GRAY2RGB)
    elif shape[-1]==4:
        img = cv2.cvtColor(img_raw_data, cv2.COLOR_RGBA2RGB)
    else:
        img = cv2.cvtColor(img_raw_data, cv2.COLOR_BGR2RGB)
    # keep min size
    max_side = max(shape[0], shape[1])
    if max_side>500:
        print('a')
        t_h = shape[0] * 500 // max_side
        t_w = shape[1] * 500 // max_side
        img =cv2.resize(img, dsize=(t_w, t_h))
    # zero mean
    img = (img - mean)  # / std  vgg paper only need means don't need std
    # # batch
    batch_img = np.expand_dims(img, axis=0)
    # flip
    # img_flip = cv2.flip(img,flipCode=1)
    # batch_img_flip = np.expand_dims(img_flip, axis=0)
    # # flip batch
    # batch_img = np.concatenate((batch_img, batch_img_flip), axis=0)
    return batch_img


def image_to_tensor(image):
    im = tf.convert_to_tensor(image)
    im = tf.expand_dims(im, axis=0)
    im_flip = tf.image.flip_left_right(im)
    im = tf.concat((im,im_flip), axis=0)
    return im


if __name__ == '__main__':
    DIR_list = [
        'D:\model_data\CUB_200_2011',
        'D:\model_data\Stanford_Dog',
        'D:\model_data\Oxford_Flower',
        'D:\model_data\Oxford_Pet',
        'D:\model_data\Aircraft'
    ]
    imf_paths, l1, im_paths, l2 = split_dataset(DIR_list[3])
    i=0
    print(len(im_paths))
    # for path,l in zip(im_paths, l2):
    im = preprocess(im_paths[3679])
    print(i)
    i+=1
        # if im.shape[-1]!=3:
        #     print(im.shape, path)
