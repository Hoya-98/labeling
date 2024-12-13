import os
import glob
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = "/home/lab/arthur/pu/data/class_images/3/B-00756_5300.jpg"
img = Image.open(image_path).convert("RGB")

plt.figure(figsize=(15,6))
plt.subplot(151)
plt.title("original_img")
plt.imshow(np.array(img))

# 원본 대미 색상 비율
ratio = 1.5
color_img = ImageEnhance.Color(img).enhance(ratio)
plt.subplot(152)
plt.title("color_img")
plt.imshow(np.array(color_img))


# 이미지 밝기 조정 0 검정 1 밝기 강함
factor = 2
brightened_img = ImageEnhance.Brightness(img).enhance(factor)

plt.subplot(153)
plt.title("brightened_img")
plt.imshow(np.array(brightened_img))


# 이미지 대비 조정 0 회색 1 강해짐
factor = 2.0
contrast_img = ImageEnhance.Contrast(img).enhance(factor)

plt.subplot(154)
plt.title("contrast_img")
plt.imshow(np.array(contrast_img))


# 선명도 조정 0 흐릿 1 강해짐
sharpness_img = ImageEnhance.Sharpness(img).enhance(5)
sharpness_img

plt.subplot(155)
plt.title("sharpness_img")
plt.imshow(np.array(sharpness_img))
