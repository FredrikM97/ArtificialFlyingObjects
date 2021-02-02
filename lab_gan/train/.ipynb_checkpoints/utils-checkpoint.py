from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import io

def model_to_json(model, path):
    model_json = model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, validation_data=None, log_dir=None):
        self.num_img = num_img
        self.validation_data = validation_data
        self.file_writer = tf.summary.create_file_writer(log_dir + "/plots/")
    
    def on_epoch_end(self, epoch, logs=None):
        real_images, last_images = next(self.validation_data)
        validate_image, fake_last_frame = self.model(real_images, training=False)
        figure = plot_sample_lastframepredictor_data_with_groundtruth(real_images * 0.5 + 0.5, last_images * 0.5 + 0.5, fake_last_frame * 0.5 + 0.5, title=f"Model Performance - Epoch: {epoch}")
        with self.file_writer.as_default():
            tf.summary.image(f"Model performance", plot_to_image(figure), step=epoch)

def plot_history(output_log_dir, d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='d-real')
    plt.plot(d2_hist, label='d-fake')
    plt.plot(g_hist, label='gen')
    plt.legend()
    # plot discriminator accuracy
    plt.subplot(2, 1, 2)
    plt.plot(a1_hist, label='acc-real')
    plt.plot(a2_hist, label='acc-fake')
    plt.legend()
    # save plot to file
    plt.savefig(output_log_dir+ '/plot_line_plot_loss.png')
    plt.show()
    plt.close()

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_sample_lastframepredictor_data_with_groundtruth(data, truth, pred, batch_size=4, show=True, title='Data Samples', save_dir=None):

    # show testing results
    fig= plt.figure(figsize=(10, 10))
    fig.suptitle(title)

    img_nbr_toshow = batch_size

    for i in range(img_nbr_toshow):
        plt.subplot(3, img_nbr_toshow, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        plt.imshow(data[i], cmap=plt.cm.binary)
        plt.xlabel("input image " + str(i))

        plt.subplot(3, img_nbr_toshow, img_nbr_toshow+ i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(truth[i], cmap=plt.cm.binary)
        plt.xlabel("truth image " + str(i))

        plt.subplot(3, img_nbr_toshow, 2*img_nbr_toshow+ i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(pred[i], cmap=plt.cm.binary)
        plt.xlabel("pred image " + str(i))
    
    if save_dir: plt.savefig(save_dir)
    #if not show: plt.close(fig) 
    #plt.show()
    return fig

def tensorboard(log_dir):
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', write_images=True)
    image_writer = tf.summary.create_file_writer(log_dir + "/images/")
    plot_writer = tf.summary.create_file_writer(log_dir + "/plots/")
    
    return tensorboard_callback, image_writer, plot_writer