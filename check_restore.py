import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os



with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('model.ckpt.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
  print(sess.run('W1:0'))
