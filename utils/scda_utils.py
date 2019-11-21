import tensorflow as tf


def non_local(feat_map):
    shape_list = feat_map.get_shape().as_list()
    # 展开hw维度 (h*w)*c的作为基础特征矩阵
    base_feat_mat = tf.reshape(feat_map, shape=(shape_list[0], shape_list[1] * shape_list[2], shape_list[3]))
    # 转置特征矩阵
    trans_feat_mat = tf.transpose(base_feat_mat, perm=[0, 2, 1])
    # 相乘 得到（h*w）*（h*w）相似矩阵
    weight_mat = tf.matmul(base_feat_mat, trans_feat_mat)
    # 每行内做softmax归一化
    norm_mat = tf.nn.softmax(weight_mat, axis=-1)
    # 与基础特征矩阵相乘 (h*w)*c的non-local特征矩阵
    nonlocal_mat = tf.matmul(norm_mat, base_feat_mat)
    # 还原h*w*c三维
    res_feature_map = tf.reshape(nonlocal_mat, shape=shape_list)
    #
    feat_map = tf.add(feat_map, res_feature_map)
    # tf.add_n()
    return feat_map


def _select_aggregate(feat_map, resize=None):
    A = tf.reduce_sum(feat_map, axis=-1, keepdims=True)  # channel维度上求和
    a = tf.reduce_mean(A, axis=[1, 2], keepdims=True)  # h\w维度上求均值
    # 可修改点了，二值m矩阵，修改位sigmod权重矩阵
    # weight_vec = tf.reshape(A, shape=[A.shape()[0], A.shape()[1]*A.shape()[2]])
    # M = tf.nn.softmax(weight_vec, axis=[1,2])
    # M = tf.reshape(M, shape=A.get_shape().as_list())

    M = tf.cast(tf.greater(A, a), dtype=tf.float32)  # 二值化mask
    # M = tf.cast(A > a, dtype=tf.float16)  # 二值化mask
    if resize is not None:
        M = tf.image.resize(M, resize)  # resize可用于查看在原图的局部定位
    return M


def scda_plus(maps, alpha=1.0):
    """    """
    map1, map2 = maps
    # map1 = non_local(map1)
    # map2 = non_local(map2)

    _, h1, w1, _ = map1.shape
    # 得到掩膜
    # M1 = _select_aggregate(map1)
    # M2 = _select_aggregate(map2)
    # # 筛选特征
    # S2 = map2 * M2
    # # 均值处理
    # pavg2 = 1.0 / tf.reduce_sum(M2, axis=[1, 2]) * tf.reduce_sum(S2, axis=[1, 2])  # (b, d)
    # # 最大值处理
    # pmax2 = tf.reduce_max(S2, axis=[1, 2])  # (b, d)
    pavg2 = tf.reduce_mean(map2, axis=[1, 2])
    pmax2 = tf.reduce_max(map2, axis=[1,2])
    # concat
    S2 = tf.concat([pavg2, pmax2], axis=-1)  # (b, 2d)
    # upsampling
    # M2 = tf.image.resize(M2, [h1, w1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # 特征筛选-处理-concat
    # S1 = map1 * (M1 * M2)  # 两层掩膜做位与运算
    # pavg1 = 1.0 / tf.reduce_sum(M1, axis=[1, 2]) * tf.reduce_sum(S1, axis=[1, 2])
    # pmax1 = tf.reduce_max(S1, axis=[1, 2])
    pavg1 = tf.reduce_mean(map1, axis=[1, 2])
    pmax1 = tf.reduce_max(map1, axis=[1,2])
    S1 = tf.concat([pavg1, pmax1], axis=-1)
    # 两层特征concat
    Splus = tf.concat([S2, S1 * alpha], axis=-1)  # (b, 4d)
    # l2 norm
    Splus = tf.nn.l2_normalize(Splus, 0)
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
