import cv2
import numpy as np


# choose codec according to format needed
def export_video(frames: list, save_path=None, fps=60):
    
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'video.avi')
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(save_path, fourcc, fps, (800, 600))

    for i in frames:
        video.write(i)

    cv2.destroyAllWindows()
    video.release()