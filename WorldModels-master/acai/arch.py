import tensorflow as tf
import tensorflow.compat.v1 as tf1

import numpy as np

CKPT_DIR = 'TEMP_CP/car_racing/ACAI_advdepth16_advweight0.5_depth16_latent32_reg0.2_scales6/tf/'
CKPT_NAME = 'model.ckpt-124250'

class ACAI:
    def __init__():
        g = tf.Graph()
        with g.as_default():
            imgs_tensor = tf1.placeholder(tf.int32, [None, 64, 64, 3], 'imgs_tensor')
            imgs_cast = tf.cast(imgs_tensor, tf.float32) * (2.0/255) - 1.0
            self.sess = tf1.Session()
            
            new_saver = tf1.train.import_meta_graph(CKPT_DIR + CKPT_NAME + '.meta')
            new_saver.restore(self.sess, CKPT_DIR + CKPT_NAME)
            
            self.encode_op = g.get_tensor_by_name("ae_enc/conv2d_12/BiasAdd:0")
            self.x = g.get_tensor_by_name("x:0")

    def encode(imgs):
        assert getattr(self, 'encode_op')
        assert getattr(self, 'x')
        imgs_input = self.sess.run(imgs_cast, feed_dict={imgs_tensor:imgs})
        encoded = self.sess.run(self.encode_op, feed_dict={self.x: imgs_input})
	img_count = encoded.shape[0]
	return np.reshape(encoded, (img_count, 32))
