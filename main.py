import os
import sys
import gzip
import time
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

from keras import Input
from keras.datasets import mnist
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
from keras.optimizers import RMSprop


def show_images(pic_dir: str):
    images = load_images(pic_dir)
    # print first 25 images
    plt.figure(1, figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


def load_images(pic_dir: str) -> [Image]:
    images_count = 10000
    original_width = 178
    original_height = 208
    diff = (original_height - original_width) // 2
    width = 64
    height = 64

    crop_rect = (0, diff, original_width, original_height - diff)
    images = []

    for pic_file in tqdm(os.listdir(pic_dir)[:images_count]):
        pic = Image.open(pic_dir + pic_file).crop(crop_rect)
        pic.thumbnail((width, height), Image.ANTIALIAS)
        images.append(np.uint8(pic))

    # Normalize the images
    images = np.array(images) / 255
    images.shape

    return images


def load_mnist():
    (X_train, y_train), (_, _) = mnist.load_data()
    f = gzip.open('/Users/denyspanov/Projects/bachelor/keras/Keras-GAN/cgan/data/mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding='bytes')
    f.close()
    (X_train, y_train), (_, _) = data

    return X_train, y_train


def create_gan(latent_dim: int, height: int, width: int, channels: int):
    generator = create_generator(latent_dim, channels)
    discriminator = create_discriminator(height, width, channels)
    discriminator.trainable = False

    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    # Adversarial Model
    optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=optimizer, loss='binary_crossentropy')

    return generator, discriminator, gan


def create_generator(latent_dim: int, channels: int) -> Model:
    gen_input = Input(shape=(latent_dim, ))

    x = Dense(64 * 8 * 8)(gen_input) # 128
    x = LeakyReLU()(x)
    x = Reshape((8, 8, 64))(x) # 128

    x = Conv2D(128, 5, padding='same')(x)  # 256
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x) # 256
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x) # 256
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x) # 256
    x = LeakyReLU()(x)

    x = Conv2D(256, 5, padding='same')(x) # 512
    x = LeakyReLU()(x)
    x = Conv2D(256, 5, padding='same')(x) # 512
    x = 4()(x)
    x = Conv2D(channels, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator


def create_discriminator(height: int, width: int, channels: int) -> Model:
    disc_input = Input(shape=(height, width, channels))

    x = Conv2D(64, 3)(disc_input) # 256
    x = LeakyReLU()(x)

    x = Conv2D(64, 4, strides=2)(x) # 256
    x = LeakyReLU()(x)

    x = Conv2D(64, 4, strides=2)(x) # 256
    x = LeakyReLU()(x)

    x = Conv2D(64, 4, strides=2)(x) # 256
    x = LeakyReLU()(x)

    x = Conv2D(64, 4, strides=2)(x) # 256
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)

    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

    return discriminator


def train(height: int, width: int, latent_dim: int, channels: int, images: np.array):
    generator, discriminator, gan = create_gan(latent_dim, height, width, channels)

    iterations = 15000
    batch_size = 16

    RES_DIR = 'res2'
    FILE_PATH = '%s/generated_%d.png'
    if not os.path.isdir(RES_DIR):
        os.mkdir(RES_DIR)

    control_size_sqrt = 6
    control_vectors = np.random.normal(size=(control_size_sqrt ** 2, latent_dim)) / 2
    start = 0
    d_losses = []
    a_losses = []
    images_saved = 0

    for step in range(iterations):
        print(f'[STEP]: {step}')
        start_time = time.time()
        latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        generated = generator.predict(latent_vectors)

        real = images[start:start + batch_size]
        combined_images = np.concatenate([generated, real])

        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        labels += .05 * np.random.random(labels.shape)

        d_loss = discriminator.train_on_batch(combined_images, labels)
        d_losses.append(d_loss)

        latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        misleading_targets = np.zeros((batch_size, 1))

        a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
        a_losses.append(a_loss)

        start += batch_size
        if start > images.shape[0] - batch_size:
            start = 0

        if step % 50 == 49:
            gan.save_weights('gan.h5')

            print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (
                step + 1, iterations, d_loss, a_loss, time.time() - start_time))

            control_image = np.zeros((width * control_size_sqrt, height * control_size_sqrt, channels))
            control_generated = generator.predict(control_vectors)

            for i in range(control_size_sqrt ** 2):
                x_off = i % control_size_sqrt
                y_off = i // control_size_sqrt
                control_image[x_off * width:(x_off + 1) * width, y_off * height:(y_off + 1) * height,
                :] = control_generated[i, :, :, :]

            im = Image.fromarray(np.uint8(control_image * 255))
            im.save(FILE_PATH % (RES_DIR, images_saved))
            images_saved += 1


if __name__ == '__main__':
    li = load_images('/Users/denyspanov/Projects/bachelor/nvidia/sources/celeba-dataset/img_align_celeba/img_align_celeba/')
    train(64, 64, 32, 3, li)
