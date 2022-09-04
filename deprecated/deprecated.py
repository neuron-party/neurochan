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

def get_score(img):
    numbers = []
    for i in range(8):
        s = img[:, (i*15):(i+1)*15]
        numbers.append(s)
    
    first_half, second_half = '', ''
    
    for i, n in enumerate(numbers):
        # first few digits in the score are affected by some weird pixel distortion, so apply higher brightness threshold
        if i < 4:
            code = get_digit_embedding(n, brightness=150)
        
            temp = [get_digit_similarity(code, i) for i in hardcode_list_2]
            predicted_digit = str(np.argmax(temp))
            first_half += predicted_digit
            
        else:
            code = get_digit_embedding(n, brightness=50)
        
            temp = [get_digit_similarity(code, i) for i in hardcode_list]
            predicted_digit = str(np.argmax(temp))
            second_half += predicted_digit
    
    full_score = first_half + second_half
    return full_score