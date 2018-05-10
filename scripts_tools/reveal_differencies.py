# -*-coding:Latin-1 -*

import numpy as np
import os, sys

from sklearn.feature_extraction import image

from scipy.ndimage import rotate
import scipy.misc
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
from scipy import ndimage
from skimage.util.shape import view_as_blocks


from PIL import Image, ImageOps

from pre_proc import *



path_gt = "../DRIVE_datasets_training_testing/test/patchs_gt/patch_gt_0_2.gif"
path_pred = "../src/preds/2_pred.png"

if not os.path.exists("./diff"):
    os.makedirs("./diff")


imgGt = Image.open(path_gt)
imgPred = Image.open(path_pred)

arrayGt = ndimage.imread(path_gt)
arrayPred = ndimage.imread(path_pred)



print(arrayPred.shape)


i = 0
j = 0

arrayGt.setflags(write = 1)

#Pre(Post)Process
for i in range(48):
    for j in range(48):
        if( arrayPred[i][j] < 100):
            arrayPred[i][j]= 0
        if(arrayPred[i][j] >= 100):
            arrayPred[i][j] = 255

for i in range(48):
    for j in range(48):
        if(arrayGt[i][j][2] != arrayPred[i][j]):
            arrayGt[i][j][2] = 0
            arrayGt[i][j][0] = 255

scipy.misc.imsave("./diff/diff"+ ".gif", arrayGt)
scipy.misc.imsave("./diff/diff2"+ ".gif", arrayPred)
