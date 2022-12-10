import tensorflow as tf
import Constants


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


# @title UNet model


def UNet():

    input_img = tf.keras.Input(
        shape=(Constants.WIDTH, Constants.HEIGHT, Constants.DEPTH))

    data_augmentation = tf.keras.Sequential(
        [tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal", seed=0),
         tf.keras.layers.experimental.preprocessing.RandomContrast(0.1, seed=0)])

    augmentation = data_augmentation(input_img)

    # ============================================================================
    # == ENCODER
    # ============================================================================

    # layer 1
    conv_1_a = tf.keras.layers.Conv2D(
        48, (3, 3), padding="same", use_bias=False)(augmentation)
    l_relu_1_a = tf.keras.layers.LeakyReLU()(conv_1_a)
    conv_1_b = tf.keras.layers.Conv2D(
        48, (3, 3), padding="same", use_bias=False)(l_relu_1_a)
    l_relu_1_b = tf.keras.layers.LeakyReLU()(conv_1_b)
    maxpool_1 = tf.keras.layers.MaxPooling2D(
        (2, 2), padding="same")(l_relu_1_b)

    # layer 2
    conv_2 = tf.keras.layers.Conv2D(
        48, (3, 3), padding="same", use_bias=False)(maxpool_1)
    l_relu_2 = tf.keras.layers.LeakyReLU()(conv_2)
    maxpool_2 = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(l_relu_2)

    # layer 3
    conv_3 = tf.keras.layers.Conv2D(
        48, (3, 3), padding="same", use_bias=False)(maxpool_2)
    l_relu_3 = tf.keras.layers.LeakyReLU()(conv_3)
    maxpool_3 = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(l_relu_3)

    # layer 4
    conv_4 = tf.keras.layers.Conv2D(
        48, (3, 3), padding="same", use_bias=False)(maxpool_3)
    l_relu_4 = tf.keras.layers.LeakyReLU()(conv_4)
    maxpool_4 = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(l_relu_4)

    # layer 5
    conv_5_a = tf.keras.layers.Conv2D(
        48, (3, 3), padding="same", use_bias=False)(maxpool_4)
    l_relu_5_a = tf.keras.layers.LeakyReLU()(conv_5_a)
    maxpool_5 = tf.keras.layers.MaxPooling2D(
        (2, 2), padding="same")(l_relu_5_a)
    conv_5_b = tf.keras.layers.Conv2D(
        48, (3, 3), padding="same", use_bias=False)(maxpool_5)
    encoded = tf.keras.layers.LeakyReLU()(conv_5_b)

    # ============================================================================
    # == DECODER
    # ============================================================================

    # layer 6
    upsamp_6 = tf.keras.layers.UpSampling2D((2, 2))(encoded)
    concat_6 = tf.concat([upsamp_6, maxpool_4], axis=3)
    conv_6_a = tf.keras.layers.Conv2D(
        96, (3, 3), padding="same", use_bias=False)(concat_6)
    l_relu_6_a = tf.keras.layers.LeakyReLU()(conv_6_a)
    conv_6_b = tf.keras.layers.Conv2D(
        96, (3, 3), padding="same", use_bias=False)(l_relu_6_a)
    l_relu_6_b = tf.keras.layers.LeakyReLU()(conv_6_b)

    # layer 7
    upsamp_7 = tf.keras.layers.UpSampling2D((2, 2))(l_relu_6_b)
    concat_7 = tf.concat([upsamp_7, maxpool_3], axis=3)
    conv_7_a = tf.keras.layers.Conv2D(
        96, (3, 3), padding="same", use_bias=False)(concat_7)
    l_relu_7_a = tf.keras.layers.LeakyReLU()(conv_7_a)
    conv_7_b = tf.keras.layers.Conv2D(
        96, (3, 3), padding="same", use_bias=False)(l_relu_7_a)
    l_relu_7_b = tf.keras.layers.LeakyReLU()(conv_7_b)

    # layer 8
    upsamp_8 = tf.keras.layers.UpSampling2D((2, 2))(l_relu_7_b)
    concat_8 = tf.concat([upsamp_8, maxpool_2], axis=3)
    conv_8_a = tf.keras.layers.Conv2D(
        96, (3, 3), padding="same", use_bias=False)(concat_8)
    l_relu_8_a = tf.keras.layers.LeakyReLU()(conv_8_a)
    conv_8_b = tf.keras.layers.Conv2D(
        96, (3, 3), padding="same", use_bias=False)(l_relu_8_a)
    l_relu_8_b = tf.keras.layers.LeakyReLU()(conv_8_b)

    # layer 9
    upsamp_9 = tf.keras.layers.UpSampling2D((2, 2))(l_relu_8_b)
    concat_9 = tf.concat([upsamp_9, maxpool_1], axis=3)
    conv_9_a = tf.keras.layers.Conv2D(
        96, (3, 3), padding="same", use_bias=False)(concat_9)
    l_relu_9_a = tf.keras.layers.LeakyReLU()(conv_9_a)
    conv_9_b = tf.keras.layers.Conv2D(
        96, (3, 3), padding="same", use_bias=False)(l_relu_9_a)
    l_relu_9_b = tf.keras.layers.LeakyReLU()(conv_9_b)

    # layer 10
    upsamp_10 = tf.keras.layers.UpSampling2D((2, 2))(l_relu_9_b)
    concat_10 = tf.concat([upsamp_10, input_img], axis=3)
    conv_10_a = tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", use_bias=False)(concat_10)
    l_relu_10_a = tf.keras.layers.LeakyReLU()(conv_10_a)
    conv_10_b = tf.keras.layers.Conv2D(
        32, (3, 3), padding="same", use_bias=False)(l_relu_10_a)
    l_relu_10_b = tf.keras.layers.LeakyReLU()(conv_10_b)

    decoded = tf.keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(
        l_relu_10_b
    )

    return tf.keras.Model(input_img, decoded)
