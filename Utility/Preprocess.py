import cv2
import matplotlib.pyplot as plt
import pathlib
import time
from Constants import Constants
import numpy as np
from tqdm import tqdm
# Preprocessing helper functions


def normalize_scan(image):
    norm_image = cv2.normalize(
        image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F  # type: ignore
    )
    return norm_image


def resize_scan(scan, desired_width, desired_height):
    scan = cv2.resize(scan, (desired_height, desired_width))
    return scan


def preprocess_scan(path, width, height):
    scan = cv2.imread(path)
    resized_scan = resize_scan(scan, width, height)
    normalized_resized_scan = normalize_scan(resized_scan)
    return normalized_resized_scan


# @title Noise helper functions


def addGaussNoise(image, mean=0, var=.01):
    row, col, ch = image.shape
    sigma = var**0.5
    np.random.seed(int(time.time()))
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


# @title Define function to preprocess and save


def preprocess_images(clean_img_dir, noise_1_dir, noise_2_dir):
    clean_img_paths = pathlib.Path(clean_img_dir).glob("*." + Constants.EXT)
    image_count = len(list(clean_img_paths))
    print(f"Preprocessing {image_count} images")
    clean_img_paths = pathlib.Path(clean_img_dir).glob("*." + Constants.EXT)
    for img_path in tqdm(clean_img_paths, total=image_count):
        img = preprocess_scan(str(img_path), Constants.WIDTH, Constants.HEIGHT)
        n_img_1 = addGaussNoise(img, Constants.MEAN, Constants.VAR)
        n_img_2 = addGaussNoise(img, Constants.MEAN, Constants.VAR)
        img_name = str(img_path).split('/')[-1]
        # print(img_name)
        cv2.imwrite(noise_1_dir + "/" + img_name, n_img_1*255)
        cv2.imwrite(noise_2_dir + "/" + img_name, n_img_2*255)
