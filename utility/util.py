import tensorflow as tf


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
