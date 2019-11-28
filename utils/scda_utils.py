import tensorflow as tf
from skimage import measure
import numpy as np


def non_local_mat(feat_map, mask):
    """
    经scda挑选出后的feat_map,做non-local运算。优化权重分配

    :param feat_map:没有batch维度
    :return:
    """
    shape_list = feat_map.shape
    # 展开hw维度 (h*w)*c的作为基础特征矩阵
    base_feat_mat = np.reshape(feat_map, newshape=(shape_list[0] * shape_list[1], shape_list[2]))
    # 转置特征矩阵
    trans_feat_mat = np.transpose(base_feat_mat, axes=[1, 0])
    # 相乘 得到（h*w）*（h*w）distance矩阵
    weight_mat = np.matmul(base_feat_mat, trans_feat_mat)
    # 按行求和，[hw]
    weight_vec = np.sum(weight_mat, axis=-1)
    # 做0均值处理，
    weight_mean = np.sum(weight_vec, axis=-1, keepdims=True) / np.sum(mask)
    weight_vec_norm = weight_vec - weight_mean
    # sigmoid函数映射为[0,1]之间的权重
    norm_mat = 1 / (1 + np.exp(-weight_vec_norm))
    # 还原shape，并利用mask过滤
    norm_map = np.reshape(norm_mat, newshape=(shape_list[0], shape_list[1], 1)) * mask

    return norm_map


def select_mask(feat_map):
    """
    feat_map 没有batch维度
    计算feature map的均值作为阈值，得到一个mask
    返回二维mask
    """
    mat = np.sum(feat_map, axis=-1)  # channel维度上求和 [h, w]
    threshold = np.mean(mat, axis=(0, 1))  # h\w维度上求均值 作为筛选阈值 [1,]
    mask = np.where(mat > threshold, 1, 0)  # 二值化mask [h, w]
    # if resize is not None:
    #     M = tf.image.resize(M, resize)  # resize可用于查看在原图的局部定位
    return mask


def max_connect(mask_map):
    """
    在二值图像上只保存最大连通区域
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
    feature map：batch=1
    pre_mask: 高层的feature map计算得到的mask [h, w, 1]， shape更小，需要放大

    返回：feature_vec 和 最大连通区域的mask
    """
    feat_map = np.squeeze(feat_map)
    # 均值
    mask = select_mask(feat_map)  # mask [height, width]
    mask = max_connect(mask)  # mask[height, width, 1]
    if pre_mask is not None:
        mask = pre_mask * mask
    select = feat_map * mask
    pavg = np.sum(select, axis=(0, 1)) / np.sum(mask)  # [channel,]
    pavg /= np.linalg.norm(pavg, keepdims=True)
    # pmax = np.max(select, axis=(0, 1))  # / np.sum(mask)
    # pmax /= np.linalg.norm(pmax, keepdims=True)
    # 最大值
    mask_weight = non_local_mat(select, mask)
    select = feat_map * mask_weight
    pmax = np.max(select, axis=(0, 1))  # / np.sum(mask)
    pmax /= np.linalg.norm(pmax, keepdims=True)
    # concat
    feat_vec = np.concatenate((pavg, pmax), axis=-1)  # (2channel,)
    # feat_vec /= np.linalg.norm(feat_vec, keepdims=True)  # 在此处进行l2-norm方便后续cos-smi计算，也可不做此步

    return feat_vec, mask


def scda_plus(maps, alpha=0.5):
    """两层SCDA特征 融合
    maps: list, 包含两个层的feature map, 第一个是pool5层，第二个是relu5_2层
    alpha: 低层特征的系数
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
    Splus = np.concatenate((feat1, feat2 * alpha), axis=-1)  # (b, 4d)
    # l2 norm
    Splus /= np.linalg.norm(Splus, keepdims=True)
    return Splus


def _nearest_neighbor(input_signal, output_size):
    '''
    最近邻插值（适用于灰度图）
    :param input_signal: 输入图像
    :param output_size: list 输出图像尺寸h,w
    :return: 缩放后的图像
    '''
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


def post_processing(feat, dim=512):
    s, u, v = tf.linalg.svd(feat)
    feat_svd = tf.transpose(v[:dim, :])  # (b, dim)?
    return feat_svd


def scda_flip_plus(batch_maps):
    """
    batch_maps 是一个list,有两个元素，分别是pool5和relu5_2层的输出
    而每个元素，是一个batch为2的feature_map
    """
    batch_pool, batch_relu = batch_maps

    # print(batch_pool[None, 1].shape)
    origin_feature = scda_plus([batch_pool[None, 0], batch_relu[None, 0]])
    flip_feature = scda_plus([batch_pool[None, 1], batch_relu[None, 1]])
    feature = np.concatenate((origin_feature, flip_feature), axis=-1)
    # feature /= np.linalg.norm(feature, keepdims=True)
    return feature


def scda_flip(batch_maps):
    """
    batch_maps: im和flip_im组成batch，得到pool5层的输出
    """
    orig_feat = batch_maps[None, 0]
    flip_feat = batch_maps[None, 1]

    origin_feature, _ = scda(orig_feat)
    flip_feature, _ = scda(flip_feat)
    feature = np.concatenate((origin_feature, flip_feature), axis=-1)
    # print(feature.shape)
    # feature /= np.linalg.norm(feature, keepdims=True)
    return feature
