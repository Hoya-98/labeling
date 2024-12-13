import PIL.Image
import numpy as np
# import cv2

from metrics import getGlobalContrastFactor, blurMetric, sharpnessMetric, exposure, GetPS, GetLS, GetLS_histogram, illuminance_uniformity
# from matlab_functions import im2double, rgb2gray, imgaborfilt

def GetQualityMetric(image_path, gray_array):
    # image_path 
    # data_type=type(gray_array[0,0])
    # num_bins = float(np.iinfo(data_type).max) - float(np.iinfo(data_type).min) + 1

    total_pixels = gray_array.shape[0] * gray_array.shape[1]
    gray_image = PIL.Image.fromarray(gray_array)
    bin_counts = gray_image.histogram()
    bin_values = np.array(range(0, len(bin_counts)))
    sum_of_pixels = np.sum(bin_counts * bin_values)
    mean_pixel = sum_of_pixels / total_pixels

    bin_value_offsets = bin_values - mean_pixel
    bin_value_offsets_sqrd = bin_value_offsets ** 2
    offset_summation = np.sum(bin_counts * bin_value_offsets_sqrd)

    s = np.sqrt(offset_summation / total_pixels)

    ImageBrightness = mean_pixel
    ImageContrast = s
    PerceivedContrast = getGlobalContrastFactor(gray_array)

    BlurScore = blurMetric(gray_array)
    SharpeScore = sharpnessMetric(gray_array)
    ExposureScore = exposure(gray_array)

    poseSym = GetPS(gray_array, 60) / 10
    lightSym = GetLS(gray_array, 60) / 10
    lightSym_hist = GetLS_histogram(gray_array, 60) / 10
    ui = illuminance_uniformity(gray_array, 60) / 10

    Qmetric = [image_path, BlurScore, SharpeScore, ExposureScore, ImageBrightness, ImageContrast, PerceivedContrast, poseSym, lightSym, lightSym_hist, ui]
    return Qmetric
