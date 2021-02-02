from .abstract_train import BaseTrain
from .utils import *
from metrics import ssim1, ssim2, psnr1, psnr2

import tensorflow as tf
from tensorflow import keras

class pix2pix_minibatches(keras.Model, BaseTrain):
    def __init__(self, discriminator, generator):
        super(pix2pix_minibatches, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
    
    def special_compile(self, 
                d_optimizer=None, 
                g_optimizer=None,
                d_loss=None,
                g_loss=None,               
                loss_fn=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
              **kwargs):
        super(pix2pix_minibatches, self).compile(metrics=metrics)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss = d_loss
        self.g_loss = g_loss
        
        
    
    @tf.function
    def train_step(self, data): 
        input_image, target = data 
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            # Generate images
            gen_output = self.generator(input_image, training=True)
            
            # Train discriminator
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)
            
            # Training
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.g_loss(disc_generated_output, gen_output, target)
            disc_loss = self.d_loss(disc_real_output, disc_generated_output)
            
            # Set weights
            generator_gradients = gen_tape.gradient(gen_total_loss,self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)
            # Update weights
            self.g_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            self.d_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
            
        #self.compiled_metrics.update_state(target, gen_output)
        
        met = {
                'gen_total_loss':gen_total_loss,
                'gen_gan_loss':gen_gan_loss,
                'gen_l1_loss':gen_l1_loss,
                'disc_loss':disc_loss, 
                
        }
        met.update({m.name: m.result() for m in self.metrics})
        return met
    
    @tf.function
    def test_step(self, data):
        real_images, last_images = data
        valid, fake_last_frame = self(real_images, training=False)

        self.compiled_metrics.update_state(last_images, fake_last_frame)
            
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def call(self, first_frame, training=False):
        fake_last_frame = self.generator(first_frame, training)
        validate_frame = self.discriminator([fake_last_frame, first_frame], training)
        
        return [validate_frame, fake_last_frame]
    
    def start_train(model, cfg, log_dir=None, train_batch_generator=None, valid_batch_generator=None, test_batch_generator=None, nbr_train_data=None, nbr_valid_data=None, nbr_test_data=None):
        cfg.BATCH_SIZE = 10
        tensorboard_callback, image_writer, plot_writer = tensorboard(log_dir)

        steps_per_epoch = (nbr_train_data // cfg.BATCH_SIZE) 
        validation_steps=(nbr_valid_data//cfg.BATCH_SIZE)
        
        generator, discriminator =  model.__model__
        loss = model.__loss__
        
        # Save the model content
        model_to_json(discriminator(), log_dir + "/discriminator.json")
        model_to_json(generator(), log_dir + "/generator.json")
        keras.utils.plot_model(model.generator(), to_file=log_dir + '/gan_gen_model.png', show_shapes=True)
        keras.utils.plot_model(model.discriminator(), to_file=log_dir + '/gan_disc_model.png', show_shapes=True)
        
        with open(log_dir+"/changes", 'a') as f:
            f.write(f"{model.__name__}\n{model.__changes__}\n{model.__train__}\n{model.__norm__}\n{model.__jitter__}")
            
        gan = pix2pix_minibatches(discriminator=discriminator(), generator=generator())
        ssim_metric = ssim1 if model.__norm__ == '[0,1]' else ssim2
        psnr_metric = psnr1 if model.__norm__ == '[0,1]' else psnr2
        
        gan.special_compile(
            d_optimizer=model.d_optimizer,
            g_optimizer=model.g_optimizer,
            d_loss=loss['d_loss_fn'],
            g_loss=loss['g_loss_fn'],
            metrics=['accuracy', ssim_metric, psnr_metric]
        )
        print("\nStarting to train model..")
        gan.fit(
            x=train_batch_generator, 
            epochs=cfg.NUM_EPOCHS, 
            verbose=1, 
            batch_size=cfg.BATCH_SIZE,
            steps_per_epoch=steps_per_epoch, #
            validation_data=valid_batch_generator,
            validation_steps=validation_steps, 
            callbacks=[GANMonitor(num_img=3, validation_data=valid_batch_generator,log_dir=log_dir), tensorboard_callback],

        ) 

        open(log_dir+"/gan_finished", 'a').close()
        
        return gan