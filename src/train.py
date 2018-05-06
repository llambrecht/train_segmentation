import keras
import os, sys
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

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

#paths
patch_nu_path = "../DRIVE_datasets_training_testing/patchs_original/"
patch_gt_path = "../DRIVE_datasets_training_testing/patchs_gt/"


DataCreator()
print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)
imgs_train, imgs_mask_train = DataLoader()

'''
imgt = Image.fromarray(imgs_train[6])
imgg = Image.fromarray(imgs_mask_train[6])
imgt.show()
imgg.show()
'''


imgs_train = preprocess(imgs_train)



imgs_mask_train = preprocess(imgs_mask_train)



imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

#imgs_train -= mean
#imgs_train /= std

imgs_mask_train = imgs_mask_train.astype('float32')
imgs_mask_train /= 255.  # scale masks to [0, 1]


'''
imgt = Image.fromarray(np.reshape(imgs_train[6], (48,48)))
imgg = Image.fromarray(np.reshape(imgs_mask_train[6], (48,48)))
imgt.show()
imgg.show()
'''




print('-'*30)
print('Creating and compiling model...')
print('-'*30)
model = get_unet(1,48,48)
model_checkpoint = ModelCheckpoint('weights.h5', monitor='loss', verbose = 1, save_best_only=True)

print('-'*30)
print('Fitting model...')
print('-'*30)
model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=numberOfEpochs, verbose=1, shuffle=True,
          validation_split=0.2,
          callbacks=[model_checkpoint])
