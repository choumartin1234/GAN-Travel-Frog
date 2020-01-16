from pix2pix import *
import os
import tensorflow as tf
import matplotlib as mpl
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(
    description='This is a pix2pix model,Reference:https://www.tensorflow.org/tutorials/generative/pix2pix')
parser.add_argument('--path', help='input file path', default='a.jpg', type=str)
args = parser.parse_args()

generator = Generator()
discriminator = Discriminator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,0.5)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

height = width = 256
def generate(model, input):
    img = tf.io.read_file(input)
    img = tf.image.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)   #load
    img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  #resize
    img = (img / 127.5) - 1  #normalize
    img = img[tf.newaxis, :]
    prediction = model(img)
    plt.imshow(prediction[0] * 0.5 + 0.5)
    plt.savefig('gen.png')

PATH = args.path
generate(generator,PATH)
