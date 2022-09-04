import numpy as np
import matplotlib.pyplot as plt


def plot_digits_mcosu(img):
    '''
    Plots each digit in the score for McOsu (score bar locations are slightly different from McOsu and osu!)
    '''
    fig, ax = plt.subplots(1, 9, figsize=(10, 10))
    for i in range(9):
        left, right = int(i * 16), int((i + 1) * 16)
        digit = img[:, start:stop]
        ax[i].imshow(digit)
    

def plot_digits_osu(img):
    fig, ax = plt.subplots(1, 8, figsize=(10, 10))
    for i in range(8):
        left, right = int(i * 15), int((i + 1) * 15)
        digit = img[:, start:stop]
        ax[i].imshow(digit)