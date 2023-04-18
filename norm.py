import pandas as pd
import numpy as np
import cv2
import statistics
from tqdm import tqdm
from glob import glob

def calculate_normalization_parameters(path=None):
    # data = pd.read_csv(path_to_train_csv)
    data = glob('NIH/images/*.png')
    mean = 0
    std = 0
    height = []
    width = []
    for i in tqdm(data):
        image = cv2.imread(i)[:, :, ::-1]
        h, w, _ = image.shape
        image = image.reshape(-1, 3)
        mean += np.mean(image, axis=0)
        std += np.std(image, axis=0)
        height.append(h)
        width.append(w)
    mean = mean / (255 * len(data))
    std = std / (255 * len(data))
    print("median height:", statistics.median(height))
    print("median width:", statistics.median(width))
    print("mean:", mean)
    print("std:", std)
    return mean, std


calculate_normalization_parameters()