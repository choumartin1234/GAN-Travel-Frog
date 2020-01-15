from pix2pix import *
checkpoint_dir='checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
generator_checkpoint=tf.train.Checkpoint(generator=generator)
generator_checkpoint.save(file_prefix=checkpoint_prefix)
