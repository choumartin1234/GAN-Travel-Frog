from pix2pix import *
checkpoint_dir='random_generator'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
generator_checkpoint=tf.train.Checkpoint(generator=generator)
generator_checkpoint.save(file_prefix=checkpoint_prefix)

