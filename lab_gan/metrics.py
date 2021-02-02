import tensorflow as tf

@tf.function
def ssim1(y_true, y_pred): # Normalised between 0,1
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)) #max_val #tf.reduce_mean(

@tf.function
def ssim2(y_true, y_pred):  # Normalised between -1,1
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0)) #max_val #tf.reduce_mean(

@tf.function
def wasserstein_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(y_true * y_pred)

@tf.function
def psnr1(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))/100

@tf.function
def psnr2(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 2.0))/100