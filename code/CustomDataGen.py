# Defining our custom data generator
import tensorflow as tf
import numpy as np
import Constants
import sys
sys.path.append("/home/pinbo/Documents/image-denoising/Constants")
sys.path.append("/home/pinbo/Documents/image-denoising/CustomDataGen")
sys.path.append("/home/pinbo/Documents/image-denoising/Utility/")


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, df,
                 batch_size=Constants.BATCH_SIZE,
                 input_size=(Constants.WIDTH,
                             Constants.HEIGHT,
                             Constants.WIDTH),
                 shuffle=True):

        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, noise_1):

        image = tf.keras.preprocessing.image.load_img(noise_1)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        return image_arr/255.

    def __get_output(self, noise_2):
        image = tf.keras.preprocessing.image.load_img(noise_2)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        return image_arr/255.

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        noise_1_batch = batches["noise_1"]
        x_batch = np.asarray([self.__get_input(noise_1)
                             for noise_1 in noise_1_batch])

        noise_2_batch = batches["noise_2"]
        y_batch = np.asarray([self.__get_output(noise_2)
                             for noise_2 in noise_2_batch])

        return x_batch, y_batch

    def __getitem__(self, index):
        # X will be a NumPy array of shape (batch_size, input_height, input_width, input_channel)
        batches = self.df[index *
                          self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size
