import tensorflow as tf

sess = tf.Session()
tf.train.import_meta_graph("/home/nightrider/Downloads/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt.meta")
tf.summary.FileWriter("__tb", sess.graph)