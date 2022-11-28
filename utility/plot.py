import cv2
import matplotlib.pyplot as plt
import pathlib
from Constants import Constants
import numpy as np
from tqdm import tqdm
# @title Helper functions to plot original noise1 and noise2 images


def generate_img(image):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def plot3(img_path, n_img_1_path, n_img_2_path):
    plt.figure(figsize=(15, 15))
    # plot clean image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    # plot noise 1
    plt.subplot(1, 3, 2)
    plt.title("Noisy Image 1")
    plt.imshow(cv2.cvtColor(cv2.imread(n_img_1_path), cv2.COLOR_BGR2RGB))
    # plot noise 2
    plt.subplot(1, 3, 3)
    plt.title("Noisy Image 2")
    plt.imshow(cv2.cvtColor(cv2.imread(n_img_2_path), cv2.COLOR_BGR2RGB))
    plt.show()


def plot_first_n(n):
    clean_img_paths = pathlib.Path(
        Constants.CLEAN_IMG_DIR).glob("*." + Constants.EXT)
    for _ in range(n):
        img_path = next(clean_img_paths)
        img_name = str(img_path).split('/')[-1]
        n_img_1 = Constants.NOISE_1_DIR + "/" + img_name
        n_img_2 = Constants.NOISE_2_DIR + "/" + img_name
        plot3(str(img_path), n_img_1, n_img_2)
