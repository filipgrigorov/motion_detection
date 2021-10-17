import cv2
import numpy as np

class MotionDet:
    def __init__(self, z_buffer_len, abs_diff_eps):
        self.z_buffer_eln = z_buffer_len
        self.abs_diff_eps = abs_diff_eps

    def detect_motion(self, depth_image):
        # check if abs diff > eps, for last N z buffer frames

        # 
