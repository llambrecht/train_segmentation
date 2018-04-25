# -*-coding:Latin-1 -*

import numpy as np
import os, sys

from scipy.ndimage import rotate
import scipy.misc
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random


from PIL import Image


#--------- Macros

nbImg = 20
height = 584
width = 565
taillePatchs = 48
nbPatchs = 50
nbPatchsPerImage = 0
channels = 3

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
rotated_original_path = dataset_path + "rotated_original/"
rotated_gt_path = dataset_path + "rotated_gt/"


#creation des répertoires
if not os.path.exists(rotated_original_path):
	os.makedirs(rotated_original_path)

if not os.path.exists(rotated_gt_path):
	os.makedirs(rotated_gt_path)



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

			open(rotated_gt_path + "rotGt" + str(i) +"_rot"+str(j) + ".tif", 'a').close()
			scipy.misc.imsave(rotated_gt_path + "rotGt" + str(i) +"_rot"+str(j) + ".gif", rotGt)








#--------- fin rotation des images et enregistrement


#--------- Création des patchs


#--------- fin création des patchs
"""
imageName = "01_test.tif"
im = Image.open(nu_imgs_test_path + imageName)
im.show()

imarray = np.array(im)
print(imarray)
"""
