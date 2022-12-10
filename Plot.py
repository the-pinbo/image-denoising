import cv2
import matplotlib.pyplot as plt
import pathlib
import Constants
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


def plot_first_n(n, clean_img_dir, noise_1_dir, noise_2_dir):
    clean_img_paths = pathlib.Path(
        clean_img_dir).glob("*." + Constants.EXT)
    for _ in range(n):
        img_path = next(clean_img_paths)
        img_name = str(img_path).split('/')[-1]
        n_img_1 = noise_1_dir + "/" + img_name
        n_img_2 = noise_2_dir + "/" + img_name
        plot3(str(img_path), n_img_1, n_img_2)

# @title Helper function to show results


def plot_history(log):
    val_loss = log.history["val_loss"]
    val_acc = log.history["val_accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax1, ax2 = axes
    ax1.plot(log.history["loss"], label="train")
    ax1.plot(val_loss, label="test")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.plot(log.history["accuracy"], label="train")
    ax2.plot(val_acc, label="test")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    for ax in axes:
        ax.legend()
