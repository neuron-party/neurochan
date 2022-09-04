import copy
import pickle
import numpy as np


def get_score(img, hardcode_list):
    numbers = []
    for i in range(8):
        s = img[:, (i*15):(i+1)*15]
        numbers.append(s)
    
    yuh = ''
    for n in numbers:
        t = n.copy()
        t[t <= 50] = 255
        code = np.where(t != 255)[0]
        
        temp = [get_digit_similarity(code, i) for i in hardcode_list]
        predicted_digit = str(np.argmax(temp))
        yuh += predicted_digit
    return yuh


def get_digit_similarity(pred, label):
    count = 0
    for i, j in zip(pred, label):
        if i == j:
            count += 1
    return count