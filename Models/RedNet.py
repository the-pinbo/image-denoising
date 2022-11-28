# Importing Modules
import tensorflow as tf
from Constants import Constants


# @title Red-Net model


def RedNet():
    """_summary_: Red-Net model


    Returns:
        tf.keras.Model: Red-Net model auto encoder with 10 layers 5 encoder and 5 decoder.
    """

    il = tf.keras.Input(
        shape=(Constants.WIDTH, Constants.HEIGHT, Constants.DEPTH))
    data_augmentation = tf.keras.Sequential(
        [tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal", seed=0),
         tf.keras.layers.experimental.preprocessing.RandomContrast(0.1, seed=0)])
    augmentation = data_augmentation(il)

    # ============================================================================
    # == ENCODER
    # ============================================================================

    l1 = tf.keras.layers.Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu",
    )(augmentation)
    l2 = tf.keras.layers.Conv2D(
        16, (3, 3), padding="same", activation="relu")(l1)
    l3 = tf.keras.layers.Conv2D(
        16, (3, 3), padding="same", activation="relu")(l2)
    l4 = tf.keras.layers.Conv2D(
        16, (3, 3), padding="same", activation="relu")(l3)
    l5 = tf.keras.layers.Conv2D(
        16, (3, 3), padding="same", activation="relu")(l4)

    # ============================================================================
    # == DECODER
    # ============================================================================

    l6 = tf.keras.layers.Conv2DTranspose(
        16, (3, 3), padding="same", activation="relu")(l5)
    s1 = tf.keras.layers.Add()([l6, l4])
    l7 = tf.keras.layers.Conv2DTranspose(
        16, (3, 3), padding="same", activation="relu")(s1)
    s2 = tf.keras.layers.Add()([l7, l3])
    l8 = tf.keras.layers.Conv2DTranspose(
        16, (3, 3), padding="same", activation="relu")(s2)
    s3 = tf.keras.layers.Add()([l8, l2])
    l9 = tf.keras.layers.Conv2DTranspose(
        16, (3, 3), padding="same", activation="relu")(s3)
    s4 = tf.keras.layers.Add()([l9, l1])
    l10 = tf.keras.layers.Conv2DTranspose(
        3, (3, 3), padding="same", activation="relu")(s4)

    model = tf.keras.Model(il, l10)
    return model
