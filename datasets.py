from __future__ import print_function


import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np





class FER2013Dataset(data.Dataset):
    def __init__(self, data_dir, train=True, imsize=48, transform=None, target_transform=None):
	
	self.data_dir = data_dir
	self.transform = transform
	self.target_transform = target_transform
	self.img_names = self.get_img_names(data_dir, train)
	self.imsize = imsize


    def get_img(self, img_path):
        img = Image.open(img_path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img




    def get_img_names(self, root, train):
	if train:
	    all_images_list_path = os.path.join(root, 'filenames_Train.txt')
	elif not train:
	    all_images_list_path = os.path.join(root, 'filenames_PublicTest.txt')
        all_images_list = np.genfromtxt(all_images_list_path, dtype=str)
        
        imgs = []

        for fname in all_images_list:
            full_path = fname
            imgs.append((full_path))#, int(fname[0:3]) - 1))
        return imgs



    def __len__(self):
	return len(self.img_names)



    def __getitem__(self, index):
	img_name = self.img_names[index]
	if img_name == 'None':
	    img_name = self.img_names[index-1]
	    print("image 'None'")
	img = self.get_img(img_name)
	labels = img_name.split("/")[5]
        return img, int(labels)

