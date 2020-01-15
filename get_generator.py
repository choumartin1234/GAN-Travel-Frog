from pix2pix import *

in_checkpoint_dir = 'checkpoints'
out_checkpoint_dir = 'app_checkpoints'
# checkpoint_prefix = os.path.join(in_checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(in_checkpoint_dir))
tf.saved_model.save(generator, out_checkpoint_dir)
# generator_checkpoint=tf.train.Checkpoint(generator=generator)
# generator_checkpoint.save(file_prefix=checkpoint_prefix)
