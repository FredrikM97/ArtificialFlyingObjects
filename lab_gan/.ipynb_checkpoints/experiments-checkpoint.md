# Testing Phase 1
https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/.
Based on https://arxiv.org/abs/1511.06434v2 a few architectural improvements have been suggested. 
* Replace ROI pooling with strided convolutions for discriminator and fractal strided convolutions for generator.
* Batchnorm on both generator and discriminator
* No fully connected layers in deeper network
* Relu in generator and output layer Tanh
* LeakyRelu as activator for discriminator in all layers
* The report also suggests that Frechet Inception Distance and inception score can be used to evaluate gan networks

# How good is my GAN?, 2018 provided different models and evaluation metrics
## Models
* Wasserstein GAN (WGAN-GP)
* SNGAN
* DCGAN - Defined as baseline generative model
* ACGAN - Not much mentioned except used with WGAN-GP
## Metrics
* Inception score
* FID
* Accuracy?

# Methods applicable on cGAN for improving performance related toimage translation application, 2019
Useless

# https://towardsdatascience.com/dcgans-generating-dog-images-with-tensorflow-and-keras-fb51a1071432
Interesting paramaters for GAN (DCGANS)
* Suggests that Weight initialization critically affect the learning process of NN
    * Example: weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=weight_init_std, mean=weight_init_mean, seed=42)
* Spectral Normalization: Improvement of Weight initialization for gan
* Instance Noise: Randomly flip labels
* Optimizers: Recommended: 0.0002 and beta:0.5 with adam
* Learning rate decay
* mode collapse
# https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py
Models for training on different architectures

# Conditional Wasserstein generative adversarial network-gradient penalty-based approach to alleviating imbalanced data classification, 2020



# Google crashcourse suggest that an accuracy of 50% chance is good since discriminator cant tell difference between images https://developers.google.com/machine-learning/gan/training


# Notes:
## What could improve:
* Better metric: FID or FJD instead of accuracy. Some problem in implementation

# TODO:
* Different number of layers
* Sigmoid vs tanh
* Model dept
* Some improvements to discriminator
* Maybe another model?
* Possible to fix blur?
* Change loss function
* add metric SSID?

## Test following (Image-to-Image Translation with conditional GAN):
* Distriminator: PatchGAN, ImageGan 
* Generator: U-Net, ResNet, ResNet-50

# Reports
Predicting Future Frames using Retrospective Cycle GAN, 2019, Yong-Hoon Kwon, Min-Gyu Park;
# Some good content:
https://gist.github.com/brannondorsey/fb075aac4d5423a75f57fbf7ccc12124
https://lijiancheng0614.github.io/2017/01/08/2017_01_08_pix2pix/

# To be mentioned in presentation
# Pix2Pix
## Model
* Skip connections
Generator:
x4 encoder: Filters: [32, 64,128,128]
conv2D + leakyReLu: Filter: 128
x4 decoder: [128,128,64,32]
## tests
* Generator last layer activation: Sigmoid vs tanh
* Normalization: -1,1 and 0,1
* Discriminator optimizer: Adam(2e-4, beta_1=0.5), SGD(learning_rate=0.01)
* Wassertein loss on generator (Missing lipschitz function therefor one cause of bad performance)
* Regularizer: L1
* Dropout: 0.5
* Kernel size: 4
* Strides: 2
* LeakyReLU: alpha=0.2
* padding='same'
* Random jitter (random resize (128,158), Random cropping, flip image on random uniform distribution)
* random_normal_initializer(0., 0.02)

## Report 
* main problem

* What is MetaGan

* what is the main contribution?

* how good are the experimental results compared to some other recent works or baselines? 

* Pros and cons 


# Fix images in tensorboard:
tensorboard --samples_per_plugin images=100