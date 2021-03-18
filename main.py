import copy
import cv2
import numpy as np

import pyrealsense2 as rs2

from collections import deque
from scipy import stats
#from skimage.transform.integral import integral_image

rgb_size = (1920, 1080)
depth_size = (1280, 720)
fps = 30

MIN_DEPTH = 300
MAX_DEPTH = 4000

THRESH = 10

def mode(frame):
    return stats.mode(frame).mode[0][0]

def estimate_depth(depth_roi, extremes=[MIN_DEPTH, MAX_DEPTH]):
    h = int(0.8 * depth_roi.shape[0])
    w = int(0.8 * depth_roi.shape[1])
    number_bins = extremes[1] - extremes[0]
    
    #frequencies: 5000 of length
    frequencies, nbins = np.histogram(depth_roi.ravel(), bins=number_bins, range=extremes)
    
    # Note: We don't want any zeros
    max_idx = np.argmax(frequencies[1:])

    return nbins[max_idx] + MIN_DEPTH

class MotionDetection:
    def __init__(self, h, w, grid_size, l=20):
        self.h = h
        self.w = w
        self.grid = grid_size

        # Devide the frame into a grid (superpixels)
        self.gh = h // grid_size
        self.gw = w // grid_size

        print(f'Grid: {self.gh} x {self.gw}')

        self.history = deque(maxlen=l)
        self.gamma = 0.9

    # If the centroid of the depth has moved, or the average depth has changed per superpixel, we have motion
    # Check moving average of the superpixel's average depth:
    # (i) If there is a lot of variance acros the history, then, it might be sporadic (depth defect)
    # (ii) If the moving average has changed relative to the previous frames, with low variance, motion perhaps???
    # (iii) Check the difference in depth values as well (absolute difference) -> After (i), (ii) and (iii) update the background

    # If we have N no motion frames, create a background
    # Motion is defined if the varinace of the motion is big enough and if the motion diff pixels themselves > threshold
    # Discount only absdiffs that meet the thresh criteria

    # Let is try to do it with point map

    def detect(self, frame):
        mask = np.zeros((self.h, self.w)).astype(np.uint8)
        depth_map = np.zeros((self.h, self.w)).astype(np.uint16)
        if len(self.history) == 0:
            for row in range(0, self.h - self.grid, self.grid):
                for col in range(0, self.w - self.grid, self.grid):
                    depth_map[row : row + self.grid, col : col + self.grid] = \
                        max(min(estimate_depth(frame[row : row + self.grid, col : col + self.grid]), MAX_DEPTH), MIN_DEPTH)

            absdiff = np.abs(depth_map - depth_map)
            self.history.append((depth_map, absdiff))

            return mask, copy.deepcopy(depth_map)

        # Construct current map
        for row in range(0, self.h - self.grid, self.grid):
            for col in range(0, self.w - self.grid, self.grid):
                depth_map[row : row + self.grid, col : col + self.grid] = \
                    max(min(estimate_depth(frame[row : row + self.grid, col : col + self.grid]), MAX_DEPTH), MIN_DEPTH)

        # Construct the absolute frame difference
        absdiff = np.abs(self.history[-1][0] - depth_map)
        self.history.append((depth_map, absdiff))

        for row in range(0, self.h - self.grid, self.grid):
            for col in range(0, self.w - self.grid, self.grid):
                depths = []
                # Make stats on the history of this superpixel (moving average and variance -> remove spurious motion detections)
                mean_depth = 0.0
                for idx in range(0, len(self.history)):
                    depths.append(estimate_depth(self.history[idx][1][row : row + self.grid, col : col + self.grid]))
                    mean_depth += (self.gamma ** idx) * depths[-1]
                mean_depth /= len(self.history)

                stddev = np.sqrt(np.sum([ (depth - mean_depth) ** 2 for depth in depths ])) / len(depths)

                mask[row : row + self.grid, col : col + self.grid] = 255 if mean_depth > THRESH and stddev < 50 else 0

        return mask, copy.deepcopy(depth_map)

if __name__ == '__main__':
    print("Start")
    pipe = rs2.pipeline()

    cfg = rs2.config()
    cfg.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, fps)
    cfg.enable_stream(rs2.stream.infrared, 640, 480, rs2.format.y8, fps)
    cfg.enable_stream(rs2.stream.depth, 640, 480, rs2.format.z16, fps)

    profile = pipe.start(cfg)

    #Intrinsics/Extrinsics:
    rgb2depth_extrin = profile.get_stream(rs2.stream.color).get_extrinsics_to(profile.get_stream(rs2.stream.depth))
    rgb_intrin = profile.get_stream(rs2.stream.color).as_video_stream_profile().get_intrinsics()
    depth_intrin = profile.get_stream(rs2.stream.depth).as_video_stream_profile().get_intrinsics()

    # Auto-alignment
    align_to = rs2.stream.color
    align = rs2.align(align_to)

    warmup_frames = 10
    print('Warming up the camera for {} frames ... \n\n'.format(warmup_frames))
    for i in np.arange(warmup_frames):
        frames = pipe.wait_for_frames()

    rows = 480
    cols = 640
    grid = 40

    det = MotionDetection(rows, cols, grid)

    while True:
        frames = pipe.wait_for_frames()
        frames = align.process(frames)

        rgb_rs = frames.get_color_frame()
        infra_rs = frames.get_infrared_frame()
        depth_rs = frames.get_depth_frame()

        rgb_np = np.ascontiguousarray(rgb_rs.as_frame().get_data(), dtype=np.uint8)
        depth_np = np.ascontiguousarray(depth_rs.as_frame().get_data(), dtype=np.uint16)

        mask, depth_map = det.detect(depth_np)
        
        # Render grid on top of the RGB
        [ cv2.line(depth_map, (0, row), (cols - 1, row), (65000, 65000, 65000), 1) for row in range(0, rows, grid) ]
        [ cv2.line(depth_map, (col, 0), (col, rows - 1), (65000, 65000, 65000), 1) for col in range(0, cols, grid) ]

        cv2.imshow('MASK', mask)
        cv2.imshow('DEPTH', depth_map.astype(np.uint16))

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()