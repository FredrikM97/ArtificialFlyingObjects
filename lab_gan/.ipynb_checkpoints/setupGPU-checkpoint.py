def setup_GPU(cfg):
    import os
    if cfg.GPU >=0: # According to https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on: Do this before importing tensorflow
        print("Creating network model using gpu " + str(cfg.GPU))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)
    elif cfg.GPU >=-1:
        print("Creating network model using cpu ")  
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    print("Finished setup of GPU")