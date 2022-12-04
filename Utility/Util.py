import tensorflow as tf
import os
import pathlib
from Constants import Constants


def download_dataset(origin_url, download_dir):
    """dataset download api

    Args:
        origin_url (_type_): _description_
        download_dir (_type_): _description_
    """
    data_root_orig = tf.keras.utils.get_file(
        origin=origin_url,
        cache_dir=download_dir,
        cache_subdir=download_dir,
        extract=True)
    print(f"Downloaded successfully in {data_root_orig}")

# @title generate image paths for noise1 and noise 2


def get_name_list(clean_img_dir):
    return [str(path).split('/')[-1] for path in pathlib.Path(clean_img_dir).glob("*." + Constants.EXT)]


def get_noise_img_path(clean_img_dir, noise_dir):
    return (noise_dir + "/" + name for name in get_name_list(clean_img_dir))


def load_model(file_name):
    autoencoder = tf.keras.models.load_model(file_name)
    print(autoencoder.summary())
    tf.keras.utils.plot_model(autoencoder, show_shapes=True, dpi=64)


def save_model(file_name, autoencoder):
    if os.path.isfile(file_name) is True:
        print("File name already exists!!!")
    else:
        autoencoder.save(file_name)
    pass


def save_tf_lite(file_name, autoencoder):
    converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
    tflite_model = converter.convert()
    with open(file_name, 'wb') as f:
        f.write(tflite_model)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Made dir: ", path)
    else:
        print(f" {path} already exists")


def get_pwd():
    return os.getcwd()


def change_dir(path):
    os.chdir(path)
    print(f"Changed directory to {get_pwd()}")


def list_dir(path):
    return os.listdir(path)
