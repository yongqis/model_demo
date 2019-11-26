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
    trans_feat_mat = np.transpose(base_feat_mat, axes=[1,0])
    # 相乘 得到（h*w）*（h*w）distance矩阵
    weight_mat = np.matmul(base_feat_mat, trans_feat_mat)
    # 按行求和，[hw]
    weight_vec = np.sum(weight_mat, axis=-1)
    # 做0均值处理，
    weight_mean = np.sum(weight_vec, axis=-1, keepdims=True) / np.sum(mask)
    weight_vec_norm = weight_vec - weight_mean
    # sigmoid函数映射为[0,1]之间的权重
    norm_mat = 1/(1+np.exp(-weight_vec_norm))
    # 还原shape，并利用mask过滤
    norm_map = np.reshape(norm_mat, newshape=(shape_list[0], shape_list[1], 1)) * mask

    return norm_map



def non_local(feat_map):
    shape_list = feat_map.shape
    # 展开hw维度 (h*w)*c的作为基础特征矩阵
    base_feat_mat = np.reshape(feat_map, newshape=(shape_list[0] * shape_list[1], shape_list[2]))
    # 转置特征矩阵
    trans_feat_mat = np.transpose(base_feat_mat, axes=[1,0])
    # 相乘 得到（h*w）*（h*w）相似矩阵
    weight_mat = np.matmul(base_feat_mat, trans_feat_mat)


    # # 按行求和，[hw]
    # weight_vec = np.sum(weight_mat, axis=-1)
    # # 做0均值处理
    # weight_mean = np.mean(weight_vec, axis=-1, keepdims=True)
    # weight_vec_norm = weight_vec - weight_mean
    # # sigmoid函数映射为[0,1]之间的权重
    # norm_mat = 1/(1+np.exp(-weight_vec_norm))
    # norm_mat = np.reshape(norm_mat, newshape=shape_list[:2])

    # 每行内做softmax归一化
    weight_vec = np.sum(weight_mat, axis=-1)  # 每行求和[hw, 1]
    weight_mat /= weight_vec # 每行相除
    # 与基础特征矩阵相乘 (h*w)*c的non-local特征矩阵
    nonlocal_mat = np.matmul(weight_mat, base_feat_mat)
    # 还原h*w*c三维
    res_feature_map = np.reshape(nonlocal_mat, newshape=shape_list)
    # 加到原feature mapfeat_map +
    feat_map = res_feature_map
    return feat_map
    # return norm_mat


def select_mask(feat_map):
    """
    feat_map 没有batch维度
    计算feature map的均值作为阈值，得到一个mask
    返回二维mask
    """
    mat = np.sum(feat_map, axis=-1)  # channel维度上求和 [h, w]
    threshold = np.mean(mat, axis=(0,1))  # h\w维度上求均值 作为筛选阈值 [1,]
    mask = np.where(mat > threshold, 1, 0) # 二值化mask [h, w]

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
    return mask


def scda(feat_map, pre_mask=None):
    """
    输入：feature map：没有batch维度
    返回：feature_vec 和 mask
    """
    # feat_map = non_local(feat_map)
    # 得到掩膜
    mask = select_mask(feat_map)  # mask [height, width]
    mask = max_connect(mask)  # mask[height, width]

    # 和高层掩膜做 与 运算
    if pre_mask:
        mask = pre_mask * mask
    feature_num = np.sum(mask)
    # 筛选特征
    mask = np.expand_dims(mask, axis=-1)  # mask[height, width, 1]
    select = feat_map * mask
    select = non_local_mat(select, mask)
    # 均值
    pavg = np.mean(select, axis=(0, 1)) # [channel,]
    # pavg /= np.linalg.norm(pavg)
    # 最大值
    pmax = np.max(select, axis=(0, 1)) # [channel,]
    # pmax /= np.linalg.norm(pmax)
    # concat
    select = np.concatenate((pavg, pmax), axis=-1)  # (2channel,)
    select /= np.linalg.norm(select, keepdims=True)

    return select, mask


def scda_plus(maps, alpha=0.5):
    """两层SCDA特征 融合"""
    map1, map2 = maps
    # map1 = non_local(map1)
    # map2 = non_local(map2)

    _, h1, w1, _ = map1.shape
    # 得到掩膜
    feat1, mask1 = scda(map1)
    # upsampling
    mask1 = tf.image.resize(mask1, [h1, w1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    feat2, _ = scda(map2, mask1)

    # 两层特征concat
    Splus = tf.concat([feat1, feat2 * alpha], axis=-1)  # (b, 4d)
    # l2 norm
    # Splus = tf.nn.l2_normalize(Splus, 0)
    return Splus


def _flip_plus(map1, map2):
    flip1 = tf.image.flip_up_down(map1)
    flip2 = tf.image.flip_up_down(map2)
    return tf.concat([scda_plus(map1, map2), scda_plus(flip1, flip2)], axis=-1)  # (b, 8d)


def post_processing(feat, dim=512):
    s, u, v = tf.linalg.svd(feat)
    feat_svd = tf.transpose(v[:dim, :])  # (b, dim)?
    return feat_svd


def scda_flip_plus(maps):
    return post_processing(_flip_plus(maps[0], maps[1]))
