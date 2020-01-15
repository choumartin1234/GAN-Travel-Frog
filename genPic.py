from pix2pix import *
import os
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt

generator = Generator()
discriminator = Discriminator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,0.5)

'''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
'''
def generate(model, input):
    img = tf.io.read_file(input)
    img = tf.image.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img[tf.newaxis, :]
    prediction = model(img)
    plt.imshow(prediction[0] * 0.5 + 0.5)
    plt.savefig('gen.png')

PATH = 'a.jpg'
generate(generator,PATH)
