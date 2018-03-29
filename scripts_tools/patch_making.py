# -*-coding:Latin-1 -*

import numpy as np
import os, sys

from PIL import Image


#--------- Macros

nbImg = 20
height = 584
width = 565
taillePatchs = 48
nbPatchs = 50
nbPatchsPerImage = 0

usage = "Usage : ./patch_making [nbPatchs]"

#--------- Paths

#training

nu_imgs_train_path = "../DRIVE/training/images/"
GT_imgs_train_path = "../DRIVE/training/1st_manual"
mask_img_train = "../DRIVE/training/mask"

#test

nu_imgs_test_path = "../DRIVE/test/images/"
GT_imgs_test_path = "../DRIVE/test/1st_manual/"
mask_img_test = "../DRIVE/test/mask"

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
print nbPatchs
print nbPatchsPerImage


#--------- fin args vars

imageName = "01_test.tif"
im = Image.open(nu_imgs_test_path + imageName)
im.show()

imarray = np.array(im)
print(imarray)
