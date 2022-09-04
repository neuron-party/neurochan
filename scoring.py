import copy
import pickle
import numpy as np


def get_score(img, hardcode_list):
    '''
    Returns the score of the current frame in an osu beatmap using pixel manipulations
    '''
    numbers = []
    for i in range(8):
        s = img[:, (i*15):(i+1)*15]
        numbers.append(s)
    
    full_score = ''
    
    for i, n in enumerate(numbers):
        code = get_digit_embedding(n, brightness=150)

        temp = [get_digit_similarity(code, i) for i in hardcode_list]
        predicted_digit = str(np.argmax(temp))
        full_score += predicted_digit
        
    return full_score


def get_digit_similarity(pred, label):
    count = 0
    for i, j in zip(pred, label):
        if i == j:
            count += 1
    return count


def get_digit_embedding(img, brightness=150):
    '''
    brightness: rgb integer (brightness parameter) for pixel filtering and embedding extraction
    '''
    d = img.copy()
    d[d <= brightness] = 255
    embedding = np.where(d != 255)[0]
    return embedding