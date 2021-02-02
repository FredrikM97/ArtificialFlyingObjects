from .abstract_model import BaseModel

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers import Concatenate, concatenate, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras.models import Model

class segmentation_model_v1(BaseModel):
    __name__='segmentation_model_v1'
    __changes__="Model based on segmentation lab"
    __train__= 'pix2pix'
    __norm__='[-1,1]'
    __jitter__= True
    
    g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    LAMBDA = 100
        
    def discriminator(self):
        def main():
            last_img = Input(shape=self.image_shape)
            first_img = Input(shape=self.image_shape)

            # Concatenate image and conditioning image by channels to produce input
            combined_imgs = Concatenate(axis=-1)([last_img, first_img])

            d1 = Conv2D(32, (3, 3), strides=(2,2), padding='same')(combined_imgs) 
            d1 = Activation('relu')(d1) 
            d2 = Conv2D(64, (3, 3), strides=(2,2), padding='same')(d1)
            d2 = Activation('relu')(d2) 
            d3 = Conv2D(128, (3, 3), strides=(2,2), padding='same')(d2)
            d3 = Activation('relu')(d3) 

            validity = Conv2D(1, (3, 3), strides=(2,2), padding='same')(d3)

            model = Model([last_img, first_img], validity)
            return model
        return main()
    
    def generator(self):
        latent_dim = 128
        def encoder(input_tensor, n_filters, kernel_size=4, strides=2,batchnorm=True, dropout=True):
            initializer = tf.random_normal_initializer(0., 0.02)
            # first layer
            x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal",
                   padding="same", strides=strides)(input_tensor)

            if batchnorm:
                x = BatchNormalization()(x)
            if dropout:
                x = Dropout(0.5)(x)
            x = LeakyReLU(alpha=0.2)(x)
            return x

        def decoder(input_tensor, skip_in,  n_filters, kernel_size=4, strides=2,batchnorm=True, dropout=True):
            initializer = tf.random_normal_initializer(0., 0.02)
            x = Conv2DTranspose(n_filters, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer=initializer)(input_tensor)
            if batchnorm:
                x = BatchNormalization()(x)
            if dropout:
                x = Dropout(0.5)(x)
            x = Concatenate()([x, skip_in])
            x = LeakyReLU(alpha=0.2)(x)
            return x

        def main():
            inputs = Input(shape=self.image_shape)

            e1 = encoder(inputs, 32, batchnorm=False)
            e2 = encoder(e1, 64)
            e3 = encoder(e2, 128)
            e4 = encoder(e3, 128)

            b = Conv2D(128, kernel_size=4, strides=2, padding='same')(e4)
            b = LeakyReLU(alpha=0.2)(b)

            u1 = decoder(b,e4, 128)
            u2 = decoder(u1,e3, 128, dropout=False)
            u3 = decoder(u2,e2, 64, dropout=False)
            u4 = decoder(u3,e1, 32, dropout=False)

            classify = Conv2DTranspose(3, kernel_size=2,  strides=(2,2), activation='tanh')(u4)
            model = Model(inputs=inputs, outputs=classify, name='Generator')
            return model
        return main()
    
    def g_loss(self, disc_generated_output, gen_output, target): # https://arxiv.org/abs/1611.07004
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss
    
    def d_loss(self, disc_real_output, disc_generated_output): # Decided by https://arxiv.org/abs/1611.07004
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss