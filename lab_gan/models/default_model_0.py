from .abstract_model import BaseModel

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers import Concatenate, concatenate, UpSampling2D
from keras.optimizers import Adam
from keras.models import Model

class default_model_v0(BaseModel):
    __name__='default_model_v0'
    __changes__="Default model given from teacher"
    
    __train__= 'pix2pix'
    __norm__='[0,1]'
    __jitter__= False
    
    g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # For loss function
    loss_object = tf.keras.losses.MeanSquaredError()
    LAMBDA = 100
    
        
    def discriminator(self):
        def main():
            last_img = Input(shape=self.image_shape)
            first_img = Input(shape=self.image_shape)

            # Concatenate image and conditioning image by channels to produce input
            combined_imgs = Concatenate(axis=-1)([last_img, first_img])

            d1 = Conv2D(32, (3, 3), strides=2, padding='same')(combined_imgs) 
            d1 = Activation('relu')(d1) 
            d2 = Conv2D(64, (3, 3), strides=2, padding='same')(d1)
            d2 = Activation('relu')(d2) 
            d3 = Conv2D(128, (3, 3), strides=2, padding='same')(d2)
            d3 = Activation('relu')(d3) 

            validity = Conv2D(1, (3, 3), strides=2, padding='same')(d3)

            model = Model([last_img, first_img], validity)
            return model
        return main()
    
    def generator(self):
        def main():

            inputs = Input(shape=self.image_shape)

            down1 = Conv2D(32, (3, 3),padding='same')(inputs)
            down1 = Activation('relu')(down1) 
            down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

            down2 = Conv2D(64, (3, 3), padding='same')(down1_pool)
            down2 = Activation('relu')(down2) 


            up1 = UpSampling2D((2, 2))(down2)
            up1 = concatenate([down1, up1], axis=3)
            up1 = Conv2D(256, (3, 3), padding='same')(up1) 
            up1 = Activation('relu')(up1) 


            up2 = Conv2D(256, (3, 3), padding='same')(up1) 
            up2 = Activation('relu')(up2) 

            nbr_img_channels = self.image_shape[2]
            outputs = Conv2D(nbr_img_channels, (1, 1), activation='sigmoid')(up2)

            model = Model(inputs=inputs, outputs=outputs, name='Generator')
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