import copy
import pickle
import numpy as np


def get_score(img, hardcode_list):
    '''
    Returns the score of current frame in an osu beatmap using pixel manipulations using a list of per-index hardcoded hashmaps rather than
    generalized index arrays, works much better for McOsu
    '''
    digits = []
    for i in range(9):
        start, stop = int(i * 16), int((i + 1) * 16)
        digit = img[:, start:stop]
        digits.append(digit)
    
    score = ''
    for i, n in enumerate(digits):
        embedding = get_digit_embedding(n)
        
        if sum(sum(sum(n <= 100))) >= 1200: # black spot, no score
            digit = '0'
        else:
            ph = hardcode_list[i] # positional hardcode hashmap
            try:
                similarities = [get_digit_similarity(embedding, ph[i]) for i in ph.keys()]
                index = np.argmax(similarities)
                digit = list(ph.keys())[index]
            except:
                # empty ph hasmap, digits in the 0-1 position are difficult to hardcode
                # such high scores are only achievable on very long beatmaps
                digit = '0'
        
        score += digit
    return score


def get_score_2(img, hardcode_list):
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