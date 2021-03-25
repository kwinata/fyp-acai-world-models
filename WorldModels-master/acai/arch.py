import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
import tensorflow.compat.v1 as tf1

import numpy as np

CKPT_DIR = 'latent16scales5/tf/'
CKPT_NAME = 'model.ckpt-48627'

INPUT_DIM = (64, 64, 3)

class ACAI:
    def __init__(self):
        self.input_dim = INPUT_DIM
        g = tf.Graph()
        with g.as_default():
            imgs_tensor = tf1.placeholder(tf.int32, [None, 64, 64, 3], 'imgs_tensor')
            imgs_cast = tf.cast(imgs_tensor, tf.float32) * (2.0/255) - 1.0
            self.sess = tf1.Session()
            
            new_saver = tf1.train.import_meta_graph(CKPT_DIR + CKPT_NAME + '.meta')
            new_saver.restore(self.sess, CKPT_DIR + CKPT_NAME)
            
            self.imgs_tensor = imgs_tensor
            self.imgs_cast = imgs_cast
            self.encode_op = g.get_tensor_by_name("ae_enc/conv2d_12/BiasAdd:0")
            self.x = g.get_tensor_by_name("x:0")

    def encode(self, imgs):
        imgs_input = self.sess.run(self.imgs_cast, feed_dict={self.imgs_tensor:imgs})
        encoded = self.sess.run(self.encode_op, feed_dict={self.x: imgs_input})
        img_count = encoded.shape[0]
        return np.reshape(encoded, (img_count, 64))
