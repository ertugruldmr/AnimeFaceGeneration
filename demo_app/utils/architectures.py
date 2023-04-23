import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing.image import array_to_img


# Declerating the configuration params
LATENT_DIM = 100
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
CHANNELS = 3

input_shape = (64, 64, 3)
alpha = 0.2


def get_generator():


    # Model architecture
    generator = Sequential(name='generator')

    # 1d random noise
    generator.add(layers.Dense(8 * 8 * 512, input_dim=LATENT_DIM))
    # model.add(layers.BatchNormalization())
    generator.add(layers.ReLU())

    # convert 1d to 3d
    generator.add(layers.Reshape((8, 8, 512)))

    # upsample to 16x16
    generator.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
    # model.add(layers.BatchNormalization())
    generator.add(layers.ReLU())

    # upsample to 32x32
    generator.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
    # model.add(layers.BatchNormalization())
    generator.add(layers.ReLU())

    # upsample to 64x64
    generator.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
    # model.add(layers.BatchNormalization())
    generator.add(layers.ReLU())

    # Output layer
    generator.add(layers.Conv2D(CHANNELS, (4, 4), padding='same', activation='tanh'))

    return generator

def get_discriminator():

    # Creating the architecture
    discriminator = Sequential(name='discriminator')

    # conv layer-1
    discriminator.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    discriminator.add(layers.BatchNormalization())
    discriminator.add(layers.LeakyReLU(alpha=alpha))
    
    # conv layer-2
    discriminator.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    discriminator.add(layers.BatchNormalization())
    discriminator.add(layers.LeakyReLU(alpha=alpha))

    # conv layer-3
    discriminator.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    discriminator.add(layers.BatchNormalization())
    discriminator.add(layers.LeakyReLU(alpha=alpha))

    # Fully Connected Classifier layer
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dropout(0.3))

    # output class
    discriminator.add(layers.Dense(1, activation='sigmoid'))

    return discriminator


class DCGAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')


    @property
    def metrics(self):
        return [self.g_loss_metric, self.d_loss_metric]
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        
    def train_step(self, real_images):
                
        # generate random noise
        batch_size = tf.shape(real_images)[0]
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # train the discriminator with real (1) and fake (0) images
        with tf.GradientTape() as tape:
            
            # discriminator (pred then loss)
            pred_real = self.discriminator(real_images, training=True)
            real_labels = tf.ones((batch_size, 1))
            real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
            d_loss_real = self.loss_fn(real_labels, pred_real)
            
            # generator (generate then implemen discriminator steps again)
            fake_images = self.generator(random_noise)

            pred_fake = self.discriminator(fake_images, training=True)
            fake_labels = tf.zeros((batch_size, 1))
            d_loss_fake = self.loss_fn(fake_labels, pred_fake)
            
            # total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            
        # compute discriminator gradients then update the gradients
        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        
        
        # train the generator model
        labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            
            # generate then implement discriminator
            fake_images = self.generator(random_noise, training=True)

            pred_fake = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(labels, pred_fake)
            
        # compute gradients then update the gradients
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        

        # update states for both models
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {'d_loss': self.d_loss_metric.result(), 'g_loss': self.g_loss_metric.result()}
    
# Custom Callback to display image per every train step
class DCGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_imgs=25, latent_dim=100):
        self.num_imgs = num_imgs
        self.latent_dim = latent_dim
        self.noise = tf.random.normal([25, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        
        # generate the image from noise
        g_img = self.model.generator(self.noise)
        
        # denormalize the image
        g_img = (g_img * 127.5) + 127.5
        g_img.numpy()
        
        # plot the image per step
        fig = plt.figure(figsize=(8, 8))
        for i in range(self.num_imgs):
            plt.subplot(5, 5, i+1)
            img = array_to_img(g_img[i])
            plt.imshow(img)
            plt.axis('off')
        # plt.savefig('epoch_{:03d}.png'.format(epoch))
        plt.show()
        
    def on_train_end(self, logs=None):
        self.model.generator.save('generator.h5')