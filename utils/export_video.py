import numpy as np
import cv2


def export_video(frames: list, save_path=None, fps=60):
    
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'video.avi')
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(save_path, fourcc, fps, (192, 144))

    for i in frames:
        video.write(i)

    cv2.destroyAllWindows()
    video.release()