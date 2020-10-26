import cv2
import numpy as np

def adaptive_thresholding(input_frame: np.ndarray):
    # blur = cv2.GaussianBlur(input_frame, (5, 5), 0)
    _, thresh = cv2.threshold(input_frame, 18, 255, cv2.THRESH_BINARY)
    return thresh
    # return cv2.adaptiveThreshold(
    #     input_frame,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     15,
    #     0,
    # )