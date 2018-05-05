import keras
import os, sys
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint

from PIL import Image
from scipy.misc import imshow

from skimage.transform import resize
from skimage.io import imsave
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras import backend as K


from data_loader import DataCreator, DataLoader, load_test_data

from keras_str import *

print('-'*30)
print('Loading and preprocessing test data...')
print('-'*30)
imgs_test, imgs_id_test = load_test_data()
imgs_test = preprocess(imgs_test)

imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('-'*30)
print('Loading saved weights...')
print('-'*30)
model.load_weights('weights.h5')

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
imgs_mask_test = model.predict(imgs_test, verbose=1)
np.save('imgs_mask_test.npy', imgs_mask_test)

print('-' * 30)
print('Saving predicted masks to files...')
print('-' * 30)
pred_dir = 'preds'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
#for image, image_id in zip(imgs_mask_test, imgs_id_test):
i=0
for image in imgs_mask_test:
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    imsave(os.path.join(pred_dir, str(i) + '_pred.png'), image)
    i+=1
