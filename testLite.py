from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import functools

tf.enable_v2_behavior()
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

PATH = './a.jpg'      #picture PATH
MPATH = './pix2pix_v1_1.0.tflite'  #Model PATH

def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.cast(img, tf.float32)
  img = img[tf.newaxis, :]
  return img


i= tf.lite.Interpreter(model_path=MPATH)
i.allocate_tensors()
input_details = i.get_input_details()
i.set_tensor(input_details[0]["index"], load_img(PATH))
i.invoke()

# Generate image.
g = i.tensor(i.get_output_details()[0]["index"])()


plt.figure(figsize=(5,5))
plt.title('Predicted')
plt.imshow(g[0] * 0.5 + 0.5)
plt.axis('off')
plt.show()
plt.savefig('pictures/test.png')

#imshow(g, 'Generated Image')