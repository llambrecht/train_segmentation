# -*-coding:Latin-1 -*

import numpy as np
import os, sys

from sklearn.feature_extraction import image

from scipy.ndimage import rotate
import scipy.misc
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random


from PIL import Image, ImageOps


#--------- Macros

nbImg = 20
height = 584
width = 565
taillePatchs = 48
nbPatchs = 50
nbPatchsPerImage = 0
channels = 3

nbPatchsTest = 5

usage = "Usage : ./patch_making [nbPatchs]"

#--------- Paths

#training

nu_imgs_train_path = "../DRIVE/training/images/"
GT_imgs_train_path = "../DRIVE/training/1st_manual/"
mask_img_train = "../DRIVE/training/mask/"

#test

nu_imgs_test_path = "../DRIVE/test/images/"
GT_imgs_test_path = "../DRIVE/test/1st_manual/"
mask_img_test = "../DRIVE/test/mask/"

#created sets
dataset_path = "../DRIVE_datasets_training_testing/"
rotated_original_path = dataset_path + "/train/rotated_original/"
rotated_gt_path = dataset_path + "/train/rotated_gt/"

#rotated image paths train
rotated_nu_path = "../DRIVE_datasets_training_testing/train/rotated_original/"
rotated_gt_path = "../DRIVE_datasets_training_testing/train/rotated_gt/"
patch_nu_path = "../DRIVE_datasets_training_testing/train/patchs_original/"
patch_gt_path = "../DRIVE_datasets_training_testing/train/patchs_gt/"

#test images path
patch_nu_path_test = "../DRIVE_datasets_training_testing/test/patchs_original/"
patch_gt_path_test = "../DRIVE_datasets_training_testing/test/patchs_gt/"

#creation des répertoires
if not os.path.exists(rotated_original_path):
	os.makedirs(rotated_original_path)

if not os.path.exists(rotated_gt_path):
	os.makedirs(rotated_gt_path)

if not os.path.exists(patch_nu_path):
	os.makedirs(patch_nu_path)

if not os.path.exists(patch_gt_path):
	os.makedirs(patch_gt_path)

if not os.path.exists(patch_nu_path_test):
	os.makedirs(patch_nu_path_test)

if not os.path.exists(patch_gt_path_test):
	os.makedirs(patch_gt_path_test)



#--------- fin Paths


#--------- Recup des arguments, def des vars


if len(sys.argv) == 2: #deux arguments
	try : #on verifie que c'est un entier
		val = int(sys.argv[1])
	except ValueError:
		print usage
	else :
		nbPatchs = int(sys.argv[1])
else:
	if len(sys.argv) != 1:
		print usage

#il faut que le nb de patchs soit un multiple
#du nombre d images
if(nbPatchs % nbImg != 0) :
	nbPatchs += nbImg-(nbPatchs%nbImg)
nbPatchsPerImage = nbPatchs/nbImg
"""print nbPatchs
print nbPatchsPerImage"""


#--------- fin args vars


#--------- Rotation des images et enregistrement

i = 0



#while (i < nbImg):
for path, subdirs, files in os.walk(nu_imgs_train_path):
	for i in range(len(files)):
		#print "original image : " + files[i]
		#on met l'image originale dans un array
		imgOriginal = Image.open(nu_imgs_train_path + files[i])
		arrayOriginal = np.asarray(imgOriginal)

		#on met la gt correspondante dans un array
		gtName = files[i][0:2] + "_manual1.gif"
		#print gtName
		gt = Image.open(GT_imgs_train_path + gtName)
		arrayGroundTruth = np.asarray(gt)

		#on prend le masque correspondant ( à faire )

		#rotation
		j = 0
		print "saving images rotations"
		for j in range(3):
			#rotation aléatoire
			rd = 1 + random.random() * 98
			rotOriginal = rotate(arrayOriginal, rd)
			rotGt = rotate(arrayGroundTruth, rd)

			open(rotated_original_path + "rotOriginal" + str(i) +"_rot"+str(j) + ".tif", 'a').close()
			scipy.misc.imsave(rotated_original_path + "rotOriginal" + str(i) +"_rot"+str(j) + ".tif", rotOriginal)

			open(rotated_gt_path + "rotGt" + str(i) +"_rot"+str(j) + ".gif", 'a').close()
			scipy.misc.imsave(rotated_gt_path + "rotGt" + str(i) +"_rot"+str(j) + ".gif", rotGt)

#--------- Création des patchs et enregistrement

#patch des images originales
for path, subdirs, files in os.walk(rotated_nu_path):
	for i in range(len(files)):
		imgOriginal = Image.open(rotated_nu_path + files[i])

			#mise en niveau de gris
		imgGris = ImageOps.grayscale(imgOriginal)

		arrayOriginal = np.asarray(imgGris)
		patches = image.extract_patches_2d(arrayOriginal, (48,48), 5)
		#mise en niveau de gris
		for j in range(len(patches)):
			open(patch_nu_path + "patch_nu_" +str(i) + "_" + str(j) + ".tif", 'a').close()
			scipy.misc.imsave(patch_nu_path + "patch_nu_" +str(i) + "_" + str(j) + ".tif", patches[j])

#patch des ground truth
for path,subdirs,files in os.walk(rotated_gt_path):
	for i in range(len(files)):
		imgGt = Image.open(rotated_gt_path + files[i])
		arrayGt = np.asarray(imgGt)
		patches = image.extract_patches_2d(arrayGt , (48,48), 5)
		for j in range(len(patches)):
			open(patch_gt_path + "patch_gt_" + str(i) + "_" + str(j) + ".gif", 'a').close()
			scipy.misc.imsave(patch_gt_path + "patch_gt_" + str(i) + "_" + str(j) + ".gif", patches[j])


#--------- fin rotation des images et enregistrement


#--------- création des patchs de test


for path, subdirs, files in os.walk(nu_imgs_test_path):
	for i in range(len(files)):
		imgOriginal = Image.open(nu_imgs_test_path + files[i])

		imgGris = ImageOps.grayscale(imgOriginal)

		arrayOriginal = np.asarray(imgGris)

		patches = image.extract_patches_2d(arrayOriginal,(48,48),5)

		for j in range(len(patches)):
			scipy.misc.imsave(patch_nu_path_test + "patch_nu_" + str(i) +"_"+str(j) +".tif" , patches[j])



for path, subdirs, files in os.walk(GT_imgs_test_path):
	for i in range(len(files)):
		imgGt = Image.open(GT_imgs_test_path + files[i])
		arrayGt = np.asarray(imgGt)
		patches = image.extract_patches_2d(arrayGt,(48,48), 5)
		for j in range(len(patches)):
			open(patch_gt_path_test	+ "patch_gt_" + str(i) + "_" + str(j) + ".gif", 'a').close()
			scipy.misc.imsave(patch_gt_path_test + "patch_gt_" + str(i) +"_"+ str(j) + ".gif", patches[j])

"""
imageName = "01_test.tif"
im = Image.open(nu_imgs_test_path + imageName)
im.show()

imarray = np.array(im)
print(imarray)
"""
