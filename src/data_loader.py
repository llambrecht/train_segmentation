import os, sys
from PIL import Image, ImageOps
import matplotlib.image as mpimg
import numpy as np



patch_nu_path = "../DRIVE_datasets_training_testing/train/patchs_original/"
patch_gt_path = "../DRIVE_datasets_training_testing/train/patchs_gt/"

patch_nu_test = "../DRIVE_datasets_training_testing/test/patchs_original/"
patch_gt_test = "../DRIVE_datasets_training_testing/test/patchs_gt/"


def DataCreator():
    images = sorted(os.listdir(patch_nu_path))
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
    images = sorted(os.listdir(patch_gt_path))

    total = len(images)
    for image_name in images:
        img = Image.open(os.path.join(patch_gt_path, image_name))
        arrayImg = np.asarray(img)
        imgs_gt[i] = arrayImg
        i += 1

    #test data
    imagestest = os.listdir(patch_nu_test)
    totaltest = len(imagestest)

    imgstest = np.ndarray((totaltest, 48,48), dtype=np.uint8)
    imgstest_gt = np.ndarray((totaltest, 48,48), dtype= np.uint8)

    #original tests
    i = 0
    for image_name in imagestest:
        imgtest = Image.open(os.path.join(patch_nu_test, image_name))
        arrayImgTest = np.asarray(imgtest)
        imgstest[i] = arrayImgTest
        i += 1

    #ground truth test
    imagestest = os.listdir(patch_gt_test)
    i=0
    for image_name in imagestest:
        imgtestGT = Image.open(os.path.join(patch_gt_test, image_name))
        arrayImgTestGt = np.asarray(imgtestGT)
        imgstest_gt[i] = arrayImgTestGt
        i+=1


    #save train data
    np.save('../imgs_train.npy', imgs)
    np.save('../imgs_gt_train.npy', imgs_gt)
    print('Saved train data to .npy files')

    #save test data
    np.save('../imgs_test.npy', imgstest)
    np.save('../imgs_gt_test.npy', imgstest_gt)
    print('Saved train data to .npy files')

def DataLoader():
    imgs_train = np.load('../imgs_train.npy')
    imgs_train_gt = np.load('../imgs_gt_train.npy')
    return imgs_train, imgs_train_gt

def load_test_data():
    imgs_test=np.load('../imgs_test.npy')
    imgs_test_gt = np.load('../imgs_gt_test.npy')
    return imgs_test, imgs_test_gt
