# @title Tensor flow gpu check
from CustomDataGen import CustomDataGen
from Constants import Constants
from Utility import Util, Plot, Preprocess
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

import pandas as pd
import tensorflow as tf


tf.test.gpu_device_name()
tf.random.set_seed(0)


DATA_ROOT = "./"
DATASET_NAME = "flickr8k"
ORIGIN_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"

DOWNLOAD_DIR = DATA_ROOT + "/dataset" + "/" + DATASET_NAME

Util.download_dataset(ORIGIN_URL, DOWNLOAD_DIR)


CLEAN_IMG_SUB_DIR = "/Flicker8k_Dataset"
CLEAN_IMG_DIR = DOWNLOAD_DIR + CLEAN_IMG_SUB_DIR
NOISE_DIR = DOWNLOAD_DIR + "/noise"

Util.make_dir(NOISE_DIR)
NOISE_DIR = NOISE_DIR + \
    f"/m_{Constants.MEAN}__v_{Constants.MEAN}".replace(".", "d")
Util.make_dir(NOISE_DIR)
NOISE_1_DIR = NOISE_DIR + "/noise_1"
Util.make_dir(NOISE_1_DIR)
NOISE_2_DIR = NOISE_DIR + "/noise_2"
Util.make_dir(NOISE_2_DIR)

Preprocess.preprocess_images(CLEAN_IMG_DIR, NOISE_1_DIR, NOISE_2_DIR)

n = 4  # @param {type:"slider", min:1, max:15, step:1}
Plot.plot_first_n(n, CLEAN_IMG_DIR, NOISE_1_DIR, NOISE_2_DIR)


noise_1_paths = pd.Series(Util.get_noise_img_path(
    CLEAN_IMG_DIR, NOISE_1_DIR), name='noise_1').astype(str)
noise_2_paths = pd.Series(Util.get_noise_img_path(
    CLEAN_IMG_DIR, NOISE_2_DIR), name='noise_2').astype(str)

Preprocess.preprocess_images(CLEAN_IMG_DIR, NOISE_1_DIR, NOISE_2_DIR)


noise_1_paths.describe()

noise_2_paths.describe()

images_df = pd.concat([noise_1_paths, noise_2_paths], axis=1).sample(
    frac=1.0, random_state=1).reset_index(drop=True)
images_df.describe()

train_size = 0.70  # @param {type:"slider", min:0, max:1, step:0.01}
train_df, test_df = train_test_split(
    images_df, train_size=train_size, shuffle=True, random_state=1)

train_df.describe()

test_df.describe()

CustomDataGen.CustomDataGen

traingen = CustomDataGen.CustomDataGen(train_df)
print(f"Number of train batches {len(traingen)}")

valgen = CustomDataGen.CustomDataGen(test_df)
print(f"Number of validation batches {len(valgen)}")


architecture = "RedNet"  # @param ["RedNet", "UNet"]
autoencoder = eval(architecture)()

autoencoder_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9)
autoencoder.compile(
    optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["accuracy"])


VERSION = "unet_1"  # @param {type:"string"}


checkpoint_path = "my_model/gauss" + \
    f"/m_{Constants.MEAN}__v_{Constants.VAR}".replace(
        ".", "d") + f"_version_{VERSION}"
monitor = "val_accuracy"
mode = "max"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor=monitor,
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode=mode,
)

# @title Train the model
EPOCHS = 15
history = autoencoder.fit(traingen,
                          validation_data=valgen,
                          epochs=EPOCHS,
                          shuffle=True,
                          callbacks=[model_checkpoint])
Plot.plot_history(history)

FILE_NAME = "./models/" + f"/m_{Constants.MEAN}__v_{Constants.VAR}".replace(
    ".", "d") + f"_version_{VERSION}" + DATASET_NAME + ".h"

TF_LITE_NAME = "models_tf_lite/" + \
    f"/m_{Constants.MEAN}__v_{Constants.VAR}".replace(".", "d") + \
    f"_version_{VERSION}" + DATASET_NAME + ".h"


Util.save_model(FILE_NAME, autoencoder)
