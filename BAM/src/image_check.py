import os
import glob
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


from matlab_functions import rgb2gray
import PIL.Image
from GetQualityMetric import GetQualityMetric


file_path = "/home/lab/arthur/pu/data/class_images/*/*.jpg"
file_list = glob.glob(file_path)
ISO_metrics = []


for image_path in file_list:
    cropped_img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    rgb = PIL.Image.fromarray(cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)).resize((512, 512),PIL.Image.ANTIALIAS)
    rgb_array=np.array(rgb)
    gray_array = cv2.equalizeHist(rgb2gray(rgb_array))
    Qmetrics = [GetQualityMetric(image_path, gray_array)]
    ISO_metrics.append(Qmetrics[0])
    print(f"{image_path} Complete")


headers = ["image_path", 'Blur_Score','Sharpe_Score','Exposure_Score','Image_Brightness','Image_Constrast','Perceived_Contrast','Pose_Symmetry','Light_Symmetry','Light_Symmetry_Histogram','Illumination_Uniformity']
ISO_extracted_features = pd.DataFrame(columns=headers, data = ISO_metrics)
ISO_extracted_features.to_csv('ISO_extracted_features.csv')