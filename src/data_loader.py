import os, sys
from PIL import Image, ImageOps
import matplotlib.image as mpimg
import numpy as np



patch_nu_path = "../DRIVE_datasets_training_testing/patchs_original/"
patch_gt_path = "../DRIVE_datasets_training_testing/patchs_gt/"


def DataCreator():
    images = os.listdir(patch_nu_path)
    total = len(images)


    imgs = np.ndarray((total,48,48), dtype=np.uint8)
    imgs_gt = np.ndarray((total, 48,48), dtype = np.uint8)

    i = 0

    #original data
    for image_name in images:
        img = Image.open(os.path.join(patch_nu_path, image_name))
        arrayImg = np.asarray(img)
        imgs[i] = arrayImg
        i += 1

    #ground truth
    i = 0
    images = os.listdir(patch_gt_path)
    total = len(images)
    for image_name in images:
        img = Image.open(os.path.join(patch_gt_path, image_name))
        arrayImg = np.asarray(img)
        imgs_gt[i] = arrayImg
        i += 1

    np.save('../imgs_train.npy', imgs)
    np.save('../imgs_gt_train.npy', imgs_gt)
    print('Saved data to .npy files')


def DataLoader():
    imgs_train = np.load('../imgs_train.npy')
    imgs_train_gt = np.load('../imgs_gt_train.npy')
    return imgs_train, imgs_train_gt


    #groundtruth data
