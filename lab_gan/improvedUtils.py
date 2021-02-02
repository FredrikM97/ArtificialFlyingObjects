# Sort images
# Get length of images
# Get sequence of images
# 
from glob import glob
import os
import tensorflow as tf
import numpy as np
import keras
import pprint
from datetime import datetime
import pytz
from utilsGAN import *
#from configGAN import flying_objects_config
#cfg = flying_objects_config()
import io
import keras.backend as kb

tz = pytz.timezone('Europe/Stockholm')

def normalize(input_image, real_image):
    # Normalize between -1 and 1
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

def normalize2(input_image, real_image):
    # Normalize between -1 and 1
    input_image = input_image / 255
    real_image = real_image / 255

    return input_image, real_image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image, hight, width):
    # Image augmentation
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, hight, width, 3])

    return cropped_image[0], cropped_image[1]

def load(first_file, last_file):
    image = tf.io.read_file(first_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    
    gt_image = tf.io.read_file(last_file)
    gt_image = tf.image.decode_jpeg(gt_image)
    gt_image = tf.cast(gt_image, tf.float32)
    return image, gt_image

def random_jitter(input_image, real_image, hight, width): # This is covered in  https://arxiv.org/abs/1611.07004v3
    # In order to prevent overfit
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, hight+30,width+30)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image, hight, width)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(first_file, last_file, image_shape, normer, jitter=True, **kwargs):
    input_image, real_image = load(first_file, last_file)
    
    if jitter:
        input_image, real_image = random_jitter(input_image, real_image, image_shape[0], image_shape[1])
    else:
        input_image, real_image = resize(input_image, real_image, image_shape[0],image_shape[1])
    input_image, real_image = normer(input_image, real_image)
    
    input_image = np.asarray(input_image)
    real_image = np.asarray(real_image)
        
    return input_image, real_image

def load_image_test(first_file, last_file, image_shape, normer, **kwargs):
    input_image, real_image = load(first_file, last_file)
    input_image, real_image = resize(input_image, real_image, image_shape[0], image_shape[1])
    input_image, real_image = normer(input_image, real_image)
    
    input_image = np.asarray(input_image)
    real_image = np.asarray(real_image)
    
    return input_image, real_image

def modded_batch_generator(data_folder, image_shape, batch_size, batch_type='train', normalize_type='[-1,1]', jitter=True):
    normer = {
        '[-1,1]':normalize,
        '[0,1]':normalize2,
    }[normalize_type]
    loader = {
        'train':load_image_train,
        'test':load_image_test,
        'validate':load_image_test
    }[batch_type]
    
    images = sorted(glob(os.path.join(data_folder, 'image', '*.png')))
    n_image = len(images)

    # first get the sequence lists
    action_dict = {}
    label_dict = {}

    for i in range(n_image):
        path, img_name = os.path.split(images[i])
        fn, ext = img_name.split(".")
        names = fn.split("_")
        action_id = int(names[0])
        class_id = names[1]
        color_id = names[2]
        frame_id = int(names[3])

        action_dict[action_id] = frame_id
        label_dict[action_id] = class_id + '_' + color_id
        
    sequence_list = []
    lastFrame_list = []
    for i in range(1,len(action_dict)+1):
        #print (action_dict.get(i))
        total_sequence_nbr = action_dict.get(i)
        for j in range(1,total_sequence_nbr+1):
            curr_list = []
            last_frame = ""
            ac_id='%06d' % i
            fr_id='%06d' % j
            lastfr_id='%06d' % action_dict.get(i)
            curr_name = ac_id+ '_' + label_dict.get(i) + '_' + fr_id+'.png'
            last_name = ac_id+ '_' + label_dict.get(i) + '_' + lastfr_id+'.png'
            file_name = os.path.join(data_folder, 'image', curr_name)
            last_frame = os.path.join(data_folder, 'image', last_name)
            curr_list.append(file_name)
            sequence_list.append(curr_list)
            lastFrame_list.append(last_frame)

    n_sequence = len(sequence_list)
    
    # this line is just to make the generator infinite, keras needs that
    while True:

        # Randomize the indices to make an array
        indices_arr = np.random.permutation(n_sequence)
        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            x_train = []  
            y_train = []

            for i in current_batch:

                image_file = sequence_list[i][0]
                last_image = lastFrame_list[i]
                
                image, gt_image = loader(image_file, last_image, image_shape, normer, jitter=jitter)
                
                # Appending them to existing batch
                x_train.append(image)
                y_train.append(gt_image)
                
            batch_images = np.array(x_train)
            batch_lables = np.array(y_train)
            # normalize image data (not the labels)
            #batch_images = batch_images.astype('float32') / 255
            #batch_lables = batch_lables.astype('float32') / 255
                
            # 2x−minxmaxx−minx−1
            yield (batch_images, batch_lables)
            
def preprocess(image_shape, normalize_type='[-1,1]', jitter=True, cfg=None):
    nbr_train_data = get_dataset_size(cfg.training_data_dir)
    nbr_valid_data = get_dataset_size(cfg.validation_data_dir)
    nbr_test_data = get_dataset_size(cfg.testing_data_dir)
    #train_batch_generator = generate_lastframepredictor_batches(cfg.training_data_dir, image_shape, cfg.BATCH_SIZE)
    #valid_batch_generator = generate_lastframepredictor_batches(cfg.validation_data_dir, image_shape, cfg.BATCH_SIZE)
    #test_batch_generator = generate_lastframepredictor_batches(cfg.testing_data_dir, image_shape, cfg.BATCH_SIZE)

    train_batch_generator = modded_batch_generator(cfg.training_data_dir, image_shape, cfg.BATCH_SIZE, batch_type='train', normalize_type=normalize_type, jitter=jitter)
    valid_batch_generator = modded_batch_generator(cfg.validation_data_dir, image_shape, cfg.BATCH_SIZE, batch_type='test', normalize_type=normalize_type)
    test_batch_generator = modded_batch_generator(cfg.testing_data_dir, image_shape, cfg.BATCH_SIZE, batch_type='validate', normalize_type=normalize_type)
    
    if cfg.DEBUG_MODE:
        t_x, t_y = next(train_batch_generator)
        print('train_x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
        print('train_y', t_y.shape, t_y.dtype, t_y.min(), t_y.max()) 
        pprint.pprint (cfg)
    
    return train_batch_generator, valid_batch_generator, test_batch_generator, nbr_train_data,nbr_valid_data, nbr_test_data
        
def logger(name):
    output_log_dir = "./logs/{}.{}".format(datetime.now(tz).strftime("%Y%m%d-%H%M%S"), name)
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)
    return output_log_dir

