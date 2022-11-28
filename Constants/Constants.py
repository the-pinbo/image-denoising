# Batch size
BATCH_SIZE = 64
WIDTH = 256  # @param {type:"number"}
HEIGHT = 256  # @param {type:"number"}
DEPTH = 3  # @param {type:"number"}
MEAN = 0  # @param {type:"number"}
VAR = .01  # @param {type:"number"}
EXT = "jpg"  # @param ["jpg", "png"] {allow-input: true}

DOWNLOAD_DIR = "./"
# @title Define noise dir and clean img dir
# @param ["/Flicker8k_Dataset"] {allow-input: true}
CLEAN_IMG_SUB_DIR = "/Flicker8k_Dataset"
CLEAN_IMG_DIR = DOWNLOAD_DIR + CLEAN_IMG_SUB_DIR
NOISE_DIR = DOWNLOAD_DIR + "/noise"
NOISE_DIR = NOISE_DIR + f"/m_{MEAN}__v_{VAR}".replace(".", "d")
NOISE_1_DIR = NOISE_DIR + "/noise_1"
NOISE_2_DIR = NOISE_DIR + "/noise_2"
