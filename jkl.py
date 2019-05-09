#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

a = tf.constant([[1, 2, 36, 8], [3, 23, 7, 2]])

c = tf.constant([[1,2,3, 4], [1,2,3, 4]])
d = a*c
sess = tf.Session()
print(sess.run(d))
