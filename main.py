import os
import time

import numpy as np

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

from keras import Input
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


def load_images(height: int, width: int, pic_dir: str) -> [Image]:
    images_count = 10000
    original_width = 178
    original_height = 208
    diff = (original_height - original_width) // 2

    crop_rect = (0, diff, original_width, original_height - diff)
    images = []

    for pic_file in tqdm(os.listdir(pic_dir)[:images_count]):
        pic = Image.open(pic_dir + pic_file).crop(crop_rect)
        pic.thumbnail((width, height), Image.ANTIALIAS)
        images.append(np.uint8(pic))

    # Normalize the images
    images = np.array(images) / 255
    # images.shape

    return images


def create_gan(latent_dim: int, height: int, width: int, channels: int,
               multiplier=2):
    generator = create_generator(latent_dim, channels, height, multiplier)
    discriminator = create_discriminator(height, width, channels, multiplier)
    discriminator.trainable = False

    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    # Adversarial Model
    optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=optimizer, loss='binary_crossentropy')

    return generator, discriminator, gan


def create_generator(latent_dim: int, channels: int, size: int,
                     multiplier=2) -> Model:
    gen_input = Input(shape=(latent_dim,))

    print(f'\nINDEX:  [{int(size / 8)}]')
    print(f'START:  [{size}]')
    print(f'MIDDLE: [{size * multiplier}]')
    print(f'END:    [{size * (multiplier * 2)}]\n')

    x = Dense(size * int(size / 8) * int(size / 8))(gen_input)  # 128
    x = LeakyReLU()(x)
    x = Reshape((int(size / 8), int(size / 8), size))(x)  # 128

    x = Conv2D(size, 5, padding='same')(x)  # 256
    x = LeakyReLU()(x)

    x = Conv2DTranspose(size, 4, strides=2, padding='same')(x)  # 256
    x = LeakyReLU()(x)

    x = Conv2DTranspose(size * multiplier, 4, strides=2, padding='same')(x)  # 256
    x = LeakyReLU()(x)

    x = Conv2DTranspose(size * multiplier, 4, strides=2, padding='same')(x)  # 256
    x = LeakyReLU()(x)

    x = Conv2D(size * (multiplier * 2), 5, padding='same')(x)  # 512
    x = LeakyReLU()(x)
    x = Conv2D(size * (multiplier * 2), 5, padding='same')(x)  # 512
    x = LeakyReLU()(x)
    x = Conv2D(channels, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator


def create_discriminator(height: int, width: int, channels: int,
                         multiplier=2) -> Model:
    disc_input = Input(shape=(height, width, channels))

    x = Conv2D(height * multiplier, 3)(disc_input)  # 256
    x = LeakyReLU()(x)

    x = Conv2D(height * multiplier, 4, strides=2)(x)  # 256
    x = LeakyReLU()(x)

    x = Conv2D(height * multiplier, 4, strides=2)(x)  # 256
    x = LeakyReLU()(x)

    x = Conv2D(height * multiplier, 4, strides=2)(x)  # 256
    x = LeakyReLU()(x)

    x = Conv2D(height * multiplier, 4, strides=2)(x)  # 256
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)

    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

    return discriminator


def train(height: int, width: int, latent_dim: int, channels: int, images: np.array,
          multiplier=2, batch_size=16, iterations=15000):
    generator, discriminator, gan = create_gan(latent_dim, height, width, channels, multiplier)

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

        print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (
            step + 1, iterations, d_loss, a_loss, time.time() - start_time))

        if step % 50 == 49:
            gan.save_weights('gan.h5')

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
    WIDTH = 64
    HEIGHT = 64
    MULTIPLIER = 4
    BATCH_SIZE = 32
    ITERATIONS = 10000
    DATA_PATH = '/home/danya_paramonov/gan/dcgan-transfer/shorten/'

    li = load_images(HEIGHT, WIDTH, DATA_PATH)
    train(HEIGHT, WIDTH, 32, 3, li, MULTIPLIER, BATCH_SIZE, ITERATIONS)
