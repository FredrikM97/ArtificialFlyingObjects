from .abstract_model import BaseModel

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers import Concatenate, concatenate
from keras.optimizers import Adam
from keras.models import Model

class Pix2Pix_model_v9(BaseModel):
    __name__='Pix2Pix_model_v9'
    __changes__="Changed to a pix2pix model in order to test a greater network for generator and discriminator. Generator: U-Net, Discriminator: PatchGAN. beta_1 to 0 and b2 to 0.9 - According to Improved Training of Wasserstein GANs, 2017. Changed learning rate on Adam" 
    __train__= 'pix2pix'
    __norm__='[-1,1]'
    __jitter__= True
    
    g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.0, beta_2=0.9)
    
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    LAMBDA = 100
        
    def discriminator(self):
        def downsample(filters, size, apply_batchnorm=True):
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential()
            result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))

            if apply_batchnorm:
                result.add(tf.keras.layers.BatchNormalization())

            result.add(tf.keras.layers.LeakyReLU())

            return result
        
        def main():
            
            initializer = tf.random_normal_initializer(0., 0.02)

            inp = tf.keras.layers.Input(shape=self.image_shape, name='input_image')
            tar = tf.keras.layers.Input(shape=self.image_shape, name='target_image')

            x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

            down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
            down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
            down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

            zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
            conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

            batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

            leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

            zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

            last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

            return tf.keras.Model(inputs=[inp, tar], outputs=last)
        
        
        return main()
    
    def generator(self):
        def downsample(filters, size, apply_batchnorm=True):
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential()
            result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))

            if apply_batchnorm:
                result.add(tf.keras.layers.BatchNormalization())

            result.add(tf.keras.layers.LeakyReLU())

            return result
        
        def upsample(filters, size, apply_dropout=False):
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential()
            result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

            result.add(tf.keras.layers.BatchNormalization())

            if apply_dropout:
                result.add(tf.keras.layers.Dropout(0.5))

            result.add(tf.keras.layers.ReLU())

            return result

        def main():
            inputs = tf.keras.layers.Input(shape=self.image_shape)

            down_stack = [
                downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
                downsample(128, 4), # (bs, 64, 64, 128)
                downsample(256, 4), # (bs, 32, 32, 256)
                downsample(512, 4), # (bs, 16, 16, 512)
                downsample(512, 4), # (bs, 8, 8, 512)
                downsample(512, 4), # (bs, 4, 4, 512)
                downsample(512, 4), # (bs, 2, 2, 512)
                #downsample(512, 4), # (bs, 1, 1, 512)
            ]

            up_stack = [
                upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
                upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
                upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
                upsample(512, 4), # (bs, 16, 16, 1024)
                upsample(256, 4), # (bs, 32, 32, 512)
                upsample(128, 4), # (bs, 64, 64, 256)
                upsample(64, 4), # (bs, 128, 128, 128)
            ]

            initializer = tf.random_normal_initializer(0., 0.02)
            last = tf.keras.layers.Conv2DTranspose(3, 4,
                                                 strides=2,
                                                 padding='same',
                                                 kernel_initializer=initializer,
                                                 activation='tanh') # (bs, 256, 256, 3)

            x = inputs

            # Downsampling through the model
            skips = []
            for down in down_stack:
                x = down(x)
                skips.append(x)

            skips = reversed(skips[:-1])

            # Upsampling and establishing the skip connections
            for up, skip in zip(up_stack, skips):
                x = up(x)
                x = tf.keras.layers.Concatenate()([x, skip])

            x = last(x)

            return tf.keras.Model(inputs=inputs, outputs=x)
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