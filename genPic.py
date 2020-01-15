import pix2pix

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
def generate(model, input):
    prediction = model(input, training=True)
    plt.imshow(prediction[0] * 0.5 + 0.5)
    plt.savefig('gen.png')

PATH = 'a.jpg'
generate(generator,PATH)
