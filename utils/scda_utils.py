import tensorflow as tf
from skimage import measure
import numpy as np


def gem_pooling(feat_map, p):
    base = np.sum(feat_map, axis=(0,1))
    base_mask = np.where(base==0, 1, 0)
    base += base_mask
    feat_vec = np.power(np.sum(np.power(feat_map, p), axis=(0,1))/base,1/p)
    feat_vec /= np.linalg.norm(feat_vec, keepdims=True)
    return feat_vec


def attention_weighted_mean_pooling(feat_map):
    """
    经scda挑选出后的feat_map,做non-local运算。优化权重分配
    :param feat_map: 没有batch维度
    :return:
    """
    shape_list = feat_map.shape
    # 展开hw维度 (h*w)*c的作为基础特征矩阵
    base_feat_mat = np.reshape(feat_map, newshape=(shape_list[0] * shape_list[1], shape_list[2]))
    # 筛选出非0特征向量,并norm
    feat_list = []
    for i in range(shape_list[0] * shape_list[1]):
        if np.sum(base_feat_mat[i]) != 0:
            f = base_feat_mat[i]  # / np.linalg.norm(base_feat_mat[i])
            feat_list.append(f)
    base_feat_mat = np.array(feat_list)
    # 转置特征矩阵
    trans_feat_mat = np.transpose(base_feat_mat, axes=[1, 0])
    # 相乘 得到（h*w）*（h*w）distance矩阵
    sim_mat = np.matmul(base_feat_mat, trans_feat_mat)
    # 按行求相似度和，并softmax映射，[hw,1]
    weight_vec = np.sum(sim_mat, axis=-1, keepdims=True) / np.sum(sim_mat)
    # 特征矩阵加权
    weighted_feat_mat = base_feat_mat * weight_vec
    # mean-poolinge
    mean_vec = np.sum(weighted_feat_mat, axis=0)
    return mean_vec


def attention_weighted_max_pooling(feat_map, weighted_mode=None):
    """
    经scda挑选出后的feat_map,做non-local运算。优化权重分配
    :param feat_map: 没有batch维度
    :param weighted_mode: 权重分配策略 'soft' or 'hard'. default None is hard
    :return:
    """
    shape_list = feat_map.shape
    # 展开hw维度 (h*w)*c的作为基础特征矩阵
    base_feat_mat = np.reshape(feat_map, newshape=(shape_list[0] * shape_list[1], shape_list[2]))
    # 筛选出非0特征向量,并norm
    feat_list = []
    for i in range(shape_list[0] * shape_list[1]):
        if np.sum(base_feat_mat[i]) != 0:
            f = base_feat_mat[i]  # / np.linalg.norm(base_feat_mat[i])
            feat_list.append(f)
    base_feat_mat = np.array(feat_list)

    # 转置特征矩阵
    trans_feat_mat = np.transpose(base_feat_mat, axes=[1, 0])
    # 相乘 得到（h*w）*（h*w）distance矩阵
    sim_mat = np.matmul(base_feat_mat, trans_feat_mat)
    # 按行求相似度和，并归一化.[hw,1]
    sim_vec = np.sum(sim_mat, axis=-1, keepdims=True) # / np.sum(sim_mat) # softmax归一化?
    sim_vec /= np.max(sim_vec)  # 最大值归一化?
    # 求均值
    sim_mean = np.mean(sim_vec, keepdims=True)
    #
    if weighted_mode is 'soft':
        # soft-使用sigmoid函数放大差异,需先做0均值处理
        sim_vec_norm = sim_vec - sim_mean
        weight_vec = 1 / (1 + np.exp(-sim_vec_norm))
    else:
        # hard-阈值分配0.1权重
        weight_vec = np.where(sim_vec > sim_mean, 1, 0)
    # weighted
    weighted_feat_mat = base_feat_mat * weight_vec

    # 最大值pooling
    max_vec = np.max(weighted_feat_mat, axis=0)

    return max_vec


def non_local_mat(feat_map, mask):
    """
    经scda挑选出后的feat_map,做non-local运算。优化权重分配
    :param feat_map: 没有batch维度
    :param mask: scda get binary mask
    :return:
    """
    shape_list = feat_map.shape
    # 展开hw维度 (h*w)*c的作为基础特征矩阵
    base_feat_mat = np.reshape(feat_map, newshape=(shape_list[0] * shape_list[1], shape_list[2]))
    # 保留目标区域特征
    feat_list = []
    for i in range(shape_list[0] * shape_list[1]):
        if np.sum(base_feat_mat[i]) != 0:
            f = base_feat_mat[i]  # / np.linalg.norm(base_feat_mat[i])
            feat_list.append(f)
    base_feat_mat = np.array(feat_list)

    # 转置特征矩阵
    trans_feat_mat = np.transpose(base_feat_mat, axes=[1, 0])
    # 相乘 得到（h*w）*（h*w）相似矩阵
    sim_mat = np.matmul(base_feat_mat, trans_feat_mat)
    # 按行求相似度均值 视为当前位置的响应程度，[hw, 1]
    sim_vec = np.sum(sim_mat, axis=-1, keepdims=True)
    # 对所有相似度分布 做做0均值处理，
    sim_mean = np.mean(sim_vec, axis=-1, keepdims=True)
    # hard weight
    weight_vec = np.where(sim_vec > sim_mean, 1, 0)
    # weighted
    weighted_feat_mat = base_feat_mat * weight_vec
    # 最大值pooling
    max_vec = np.max(weighted_feat_mat, axis=0)

    return max_vec


def mean_mask(feat_map, pre_mask):
    """
    :param feat_map: 没有batch维度的feature map
    :param pre_mask: 底层计算除了的mask
    计算feature map的均值作为阈值，得到一个mask
    返回二维mask
    """
    channel_mean = np.mean(feat_map, axis=(0,1),keepdims=True) / 2
    channel_mask = np.where(feat_map>channel_mean, 1, 0)
    feat_map_select = feat_map * channel_mask
    # print(feat_map_select.shape)
    # print(np.sum(feat_map, axis=(0,1)))
    # print(np.sum(feat_map_select, axis=(0,1)))
    mat = np.sum(feat_map_select, axis=-1)  # channel维度上求和 [h, w]
    threshold = np.mean(mat, axis=(0, 1))  # h\w维度上求均值 作为筛选阈值 [1,]
    mask = np.where(mat > threshold, 1, 0)  # 二值化mask [h, w]
    # if resize is not None:
    #     M = tf.image.resize(M, resize)  # resize可用于查看在原图的局部定位
    if pre_mask is not None:
        mask = mask * pre_mask
    mask = _max_connect(mask)
    return mask


def _max_connect(mask_map):
    """
    在二值图像上求得最大连通区域

    :param mask_map: binary matrix
    :return: new mask
    """
    # 每个连通区域分配一个label
    areas_label = measure.label(mask_map, connectivity=2)
    # 每个连通区域的属性
    areas_prop = measure.regionprops(areas_label)
    # 找到最大连通区域的label
    max_areas = 0
    max_label = 0
    for sub_area in areas_prop:
        if sub_area.area > max_areas:
            max_areas = sub_area.area
            max_label = sub_area.label
    # 保留label区域
    mask = np.where(areas_label == max_label, 1, 0)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def scda(feat_map, pre_mask=None):
    """
    :param feat_map：batch=1
    :param pre_mask: 高层的feature map计算得到的mask [h, w, 1]， shape更小，需要放大
    :return: feature_vec和最大连通区域得到的mask
    """
    feat_map = np.squeeze(feat_map)  # 去掉batch

    # 求目标区域二值掩膜mask
    mask = mean_mask(feat_map, pre_mask)  # mask [height, width]
    # 筛选目标区域
    obj = feat_map * mask

    # feat_vec = gem_pooling(obj,4)  # Gem

    # 目标区域均值
    pavg = np.sum(obj, axis=(0, 1)) / np.sum(mask)  # [channel,]
    pavg /= np.linalg.norm(pavg, keepdims=True)
    # 目标区域最大值
    # pmax = np.max(obj, axis=(0, 1)),'soft'
    # 目标区域加权最大值
    pmax = attention_weighted_max_pooling(obj)
    pmax /= np.linalg.norm(pmax, keepdims=True)
    # concat
    feat_vec = np.concatenate((pavg, pmax), axis=-1)

    return feat_vec, mask


def scda_flip(batch_maps):
    """
    将origin_im和flip_im组成batch送入模型，对某一卷积层的输出进行处理

    :param batch_maps: feature_map, batch=2
    :return 两个scda特性concat shape(2048,)
    """
    orig_feat = batch_maps[None, 0]
    flip_feat = batch_maps[None, 1]

    origin_feature, _ = scda(orig_feat)
    flip_feature, _ = scda(flip_feat)
    feature = np.concatenate((origin_feature, flip_feature), axis=-1)

    return feature


def scda_plus(maps, alpha=0.5):
    """
    两层特征融合

    :params maps: list, 包含两层的feature map, 第一个是pool5层，第二个是relu5_2层
    :params alpha: 低层特征的concat系数
    :return: 两层特征concat，shape(2048,)
    """
    map1, map2 = maps
    _, h2, w2, _ = map2.shape  #
    # 得到掩膜
    feat1, mask1 = scda(map1)
    # upsampling
    mask1 = _nearest_neighbor(mask1, [h2, w2])
    mask1 = np.expand_dims(mask1, axis=-1)
    feat2, _ = scda(map2, mask1)

    # 两层特征concat
    plus = np.concatenate((feat1, feat2 * alpha), axis=-1)  # (b, 4d)
    # l2 norm
    # plus /= np.linalg.norm(plus, keepdims=True)
    return plus


def _nearest_neighbor(input_signal, output_size):
    """
    最近邻插值（适用于灰度图）

    :param input_signal: 输入图像
    :param output_size: list 输出图像尺寸h,w
    :return: 缩放后的图像
    """
    input_signal_cp = np.copy(input_signal)  # 输入图像的副本
    input_row, input_col, _ = input_signal_cp.shape  # 输入图像的尺寸（行、列）
    output_signal = np.zeros(output_size)  # 输出图片
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            # 输出图片中坐标 （i，j）对应至输入图片中的（m，n）
            m = round(i / output_size[0] * input_row)
            n = round(j / output_size[1] * input_col)
            # 防止四舍五入后越界
            if m >= input_row:
                m = input_row - 1
            if n >= input_col:
                n = input_col - 1
            # 插值
            output_signal[i, j] = input_signal_cp[m, n]
    return output_signal


def scda_flip_plus(batch_maps):
    """
    将origin_im和flip_im组成batch送入模型，对两个不同卷积层的输出进行处理

    :param batch_maps: list,分别是pool5层和relu5_2层的输出的batch为2的feature_map
    :return: feature shape (4096,)
    """
    batch_pool, batch_relu = batch_maps
    origin_feature = scda_plus([batch_pool[None, 0], batch_relu[None, 0]])
    flip_feature = scda_plus([batch_pool[None, 1], batch_relu[None, 1]])
    feature = np.concatenate((origin_feature, flip_feature), axis=-1)

    return feature


def post_processing(feat, dim=512):
    """
    svd分解
    """
    s, u, v = tf.linalg.svd(feat)
    feat_svd = tf.transpose(v[:dim, :])  # (b, dim)?
    return feat_svd
