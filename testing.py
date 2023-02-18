import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# for GPU process:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import h5py
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import logging
import losses
import pandas as pd
import cv2
from skimage import color
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.python.client import device_lib

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
assert 'GPU' in str(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('is_gpu_available: %s' % tf.test.is_gpu_available())  # True/False
# Or only check for gpu's with cuda support
print('gpu with cuda support: %s' % tf.test.is_gpu_available(cuda_only=True))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# tf.config.list_physical_devices('GPU') #The above function is deprecated in tensorflow > 2.1

def standardize_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)


def normalize_image(image):
    '''
    make image normalize between 0 and 1
    '''
    img_o = np.float32(image.copy())
    img_o = (img_o - img_o.min()) / (img_o.max() - img_o.min())
    return img_o


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log_root = 'D:\Artefact_correction\logdir'
experiment_name = 'prova1'
forceoverwrite = False
input_fold = r'D:\Artefact_correction\data'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('-' * 50)
print('Testing...')
print('-' * 50)
model_path = os.path.join(log_root, experiment_name)

logging.info('------- loading model -----------')
model = tf.keras.models.load_model(os.path.join(model_path, 'model_weights.h5'),
                                   custom_objects={'loss_function': losses.combo_loss(),
                                                   'dice_coef': losses.dice_coef})

out_fold = os.path.join(model_path, 'test')
if not tf.io.gfile.exists(out_fold) or forceoverwrite:
    try:
        shutil.rmtree(out_fold)
    except:
        pass
    tf.io.gfile.makedirs(out_fold)


for file in ['sel_art', 'sel_sani']:
    print(file)
    test_pred = []
    figs = []
    data = h5py.File(os.path.join(input_fold, file + '.hdf5'), 'r')
    dim = 192

    MASK = []
    IMG = []
    for i in range(len(data['img_sel'][()])):
        if i % 100 == 0:
            print(i)
        img = cv2.resize(data['img_sel'][i], (dim,dim), interpolation=cv2.INTER_AREA)
        IMG.append(np.float32(img))
        img = np.float32(normalize_image(img))
        x = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        mask_out = model.predict(x)
        mask_out = np.squeeze(mask_out)
        MASK.append(mask_out)

    data.close()

    hdf5_file = h5py.File(os.path.join(out_fold, file + '.hdf5'), "w")
    hdf5_file.create_dataset('mask', [len(MASK)] + [dim, dim], dtype=np.uint8)
    hdf5_file.create_dataset('img_raw', [len(IMG)] + [dim, dim], dtype=np.float32)

    for i in range(len(IMG)):
        hdf5_file['mask'][i, ...] = MASK[i]
        hdf5_file['img_raw'][i, ...] = IMG[i]

    hdf5_file.close()

    '''
    for i in range(len(data['img_sel'][()])):
    
        if i % 100 == 0:
            print(i)
            img = np.float32(normalize_image(data['img_sel'][i]))
            x = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
            mask_out = model.predict(x)
            mask_out = np.squeeze(mask_out)
            test_pred.append(mask_out)

            img_raw = cv2.normalize(src=data['img_sel'][i], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)
            fig = plt.figure(figsize=(14, 14))
            ax1 = fig.add_subplot(121)
            ax1.set_axis_off()
            ax1.imshow(img_raw.astype('uint8'), cmap='gray')
            ax2 = fig.add_subplot(122)
            ax2.set_axis_off()
            ax2.imshow(
                color.label2rgb(mask_out.astype(np.uint8), img_raw, colors=[(255, 0, 0), (0, 255, 0), (255, 255, 0)], alpha=0.0008, bg_label=0,
                                bg_color=None))
            ax1.title.set_text('Raw_img')
            ax2.title.set_text('Automated')
            txt = str('paz: ' + data['paz'][i])
            plt.text(0.1, 0.80, txt, transform=fig.transFigure, size=18)
            figs.append(fig)
            # plt.show()
    
        
    data.close()
    
    pdf_path = os.path.join(out_fold, file + '_plt_imgs.pdf')

    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close()
    '''
