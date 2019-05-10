#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os

path = 'a/b/c/66.txt'
label = os.path.split(os.path.dirname(path))
print(label)