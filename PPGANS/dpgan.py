import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import initializers

from gradient_noise import add_gradient_noise
NoisyAdam = add_gradient_noise(Adam)

K.set_image_dim_ordering('th')

np.random.seed(0) # Deterministic output.
random_dim = 100 # For consistency with other GAN implementations.

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(60000, 784)

# Generator
generator = Sequential()
generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(X_train.shape[1], activation='tanh'))

generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
generator.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

# Discriminator
discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=X_train.shape[1], kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))

clipnorm = 5.0
standard_deviation = 0.0001
discriminator_optimizer = NoisyAdam(lr=0.0002, beta_1=0.5, clipnorm=clipnorm, standard_deviation=standard_deviation)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# GAN
discriminator.trainable = False
gan_input = Input(shape=(random_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(inputs=gan_input, outputs=gan_output)

gan_optimizer = Adam(lr=0.0002, beta_1=0.5)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# Losses for plotting
discriminator_losses = []
generator_losses = []

def plot_loss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(discriminator_losses, label='Discriminitive Loss')
    plt.plot(generator_losses, label='Generative Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_{}.png'.format(epoch))

def plot_generated_images(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('images/gan_generated_image_epoch_{}.png'.format(epoch))

def save_models(epoch):
    generator.save('models/gan_generator_epoch_{}.h5'.format(epoch))
    discriminator.save('models/gan_discriminator_epoch_{}.h5'.format(epoch))

def train(epochs=1, batch_size=64):
    batch_count = int(X_train.shape[0] / batch_size)

    for e in range(1, epochs+1):
        print('-' * 15, 'Epoch {}'.format(e), '-' * 15)
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            # print np.shape(image_batch), np.shape(generated_images)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2 * batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator_loss = discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            generator_loss = gan.train_on_batch(noise, y_gen)

        # Store loss of most recent batch from this epoch
        discriminator_losses.append(discriminator_loss)
        generator_losses.append(generator_loss)

        plot_generated_images(e)
        if e == 1 or e % 20 == 0:
            save_models(e)

    # Plot losses from every epoch
    plot_loss(e)

if __name__ == '__main__':
    for path in ['images', 'models']:
        if not os.path.exists(path):
            os.makedirs(path)

    train(100, 128)

