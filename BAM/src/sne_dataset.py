#-*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image 
from PIL import ImageEnhance
from src.metrics import blurMetric, getGlobalContrastFactor
from src.matlab_functions import rgb2gray


class CustomDataset(Dataset):
    def __init__(self, root_dir, image_dir, cls_num=4, resize_length=512, set_name='data', transform=None, role = "train"):
        """
            root_dir (str) : /home/lab/arthur/pu/data

            transform (callable, optional) : optional transform to be applied on a sample..
        """
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.set_name = set_name
        self.role = role
        self.resize_length = resize_length
        self.csv = pd.read_csv(os.path.join("/data/pu/data", self.root_dir))

        self.transform = transform

        self.image_ids, self.labels, self.classes = self.load_csv()

    def load_csv(self):
        # For extra data 
        paths = self.csv["file_name"][self.csv["split"] == self.role].tolist()
        labels = self.csv["label"][self.csv["split"] == self.role].tolist()
        classes = sorted([1,2,3,4])

        return paths, labels, classes

    def load_image(self, image_index): 
        img_path = os.path.join(self.image_dir, str(self.labels[image_index]), self.image_ids[image_index]) # modified_image에서 load
        if not os.path.exists(img_path):
            print(f"Don't Exists {img_path}")
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.resize_length, self.resize_length))

        gray_array = cv2.equalizeHist(rgb2gray(np.array(img)))

        # blur_score = self.check_blur(gray_array)
        # contrast_score = self.check_contrast(gray_array)
        brightness_score = self.check_brightness(gray_array)

    
        #if blur_score >= 60:
        #    img = ImageEnhance.Sharpness(img).enhnace(5)


        #if contrast_score >= 73.77:
        #    img = ImageEnhance.Contrast(img).enhance(2.0)

        if brightness_score <= 130.1:
            img = ImageEnhance.Brightness(img).enhance(1.5)


        return img

    def load_annotations(self, image_index):
        label = self.labels[image_index]
        label = self.classes.index(label) # ++
        label = torch.tensor(label, dtype=torch.long)
        return label


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):
        sample = self.load_image(idx)
        annot = self.load_annotations(idx)

        w, h = sample.size
        cen_x = int(w // 2)
        cen_y = int(h // 2)

        # reflect pad augmentation
        # if cen_x < (self.resize_length // 2) and cen_y < (self.resize_length // 2):
        #     paddings = ((self.resize_length // 2)-cen_x, (self.resize_length // 2)-cen_y)
        #     sample = transforms.Pad(padding=paddings, padding_mode="symmetric")(sample)
        
        # elif cen_x < (self.resize_length // 2):
        #     paddings = ((self.resize_length // 2)-cen_x, 0, (self.resize_length // 2)-cen_x, 0)
        #     sample = transforms.Pad(padding=paddings, padding_mode="symmetric")(sample)
        
        # elif cen_y < (self.resize_length  //2) :
        #     paddings = (0, (self.resize_length // 2)-cen_y, 0, (self.resize_length // 2)-cen_y)
        #     sample = transforms.Pad(padding=paddings, padding_mode="symmetric")(sample)

        
        if self.transform:
            sample = self.transform(sample)

        return sample, annot


    def check_blur(self, hist_img):
        blur_score = blurMetric(hist_img)

        return blur_score

    
    def check_contrast(self, gray_array):
        total_pixels = gray_array.shape[0] * gray_array.shape[1]
        gray_image = Image.fromarray(gray_array)
        bin_counts = gray_image.histogram()
        bin_values = np.array(range(0, len(bin_counts)))
        sum_of_pixels = np.sum(bin_counts * bin_values)
        mean_pixel = sum_of_pixels / total_pixels

        bin_value_offsets = bin_values - mean_pixel
        bin_value_offsets_sqrd = bin_value_offsets ** 2
        offset_summation = np.sum(bin_counts * bin_value_offsets_sqrd)

        contrast_score = np.sqrt(offset_summation / total_pixels)

        return contrast_score


    def check_brightness(self, gray_array):
        total_pixels = gray_array.shape[0] * gray_array.shape[1]
        gray_image = Image.fromarray(gray_array)
        bin_counts = gray_image.histogram()
        bin_values = np.array(range(0, len(bin_counts)))
        sum_of_pixels = np.sum(bin_counts * bin_values)
        mean_pixel = sum_of_pixels / total_pixels
        brightness_score = mean_pixel

        return brightness_score
