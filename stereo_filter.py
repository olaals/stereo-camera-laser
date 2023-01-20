import os
from harvesters.core import simplefilter
from genicam_wrappers import GenicamStereo
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def save_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f)

def read_calib(calib_path):
    calib_dict = read_json(calib_path)
    H = np.array(calib_dict["H"])
    lK = np.array(calib_dict['left_K'])
    ldc = np.array(calib_dict['left_dist'])
    rK = np.array(calib_dict['right_K'])
    rdc = np.array(calib_dict['right_dist'])
    T_ltr = np.array(calib_dict['T_ltr'])
    return lK, ldc, rK, rdc, T_ltr, H

def apply_planar_homography(H, img):
    projected_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    return projected_img

def capture_homography(cti_path, calib_path, exposure_time, save_dir):
    os.makedirs(save_dir)
    left_save_dir = os.path.join(save_dir, 'left')
    right_save_dir = os.path.join(save_dir, 'right')
    os.makedirs(left_save_dir)
    os.makedirs(right_save_dir)
    calib_dict = read_json(calib_path)
    lK, ldc, rK, rdc, T_ltr, H = read_calib(calib_path)
    print("lK", lK)
    print("ldc", ldc)
    print("rK", rK)
    print("rdc", rdc)
    print("T_ltr", T_ltr)
    print("H",H)
    exposure_time = 30000
    device_num_left = 1
    device_num_right = 0
    cap = GenicamStereo(cti_path, device_num_left, device_num_right, exposure_time, gray=True)
    save_num = 0
    while True:
        left, right = cap.read()
        # undist left and right
        left = cv2.undistort(left, lK, ldc)
        right = cv2.undistort(right, rK, rdc)
        projected = apply_planar_homography(H, right)
        thresh = 30
        threshed = np.bitwise_and(projected>thresh,left>thresh)
        simple_filter = np.where(threshed, left, 0)
        depth_stack = np.dstack((left, projected, projected))
        cv2.imshow('left', left)
        cv2.imshow('right', right)
        cv2.imshow('projected', depth_stack)
        cv2.imshow('filtered', simple_filter)
        key = cv2.waitKey(1)
        # save on keypress s
        if key == ord('s'):
            print("saving")
            left_name = os.path.join(left_save_dir, f'left_{save_num}.png')
            right_name = os.path.join(right_save_dir, f'right_{save_num}.png')
            cv2.imwrite(left_name, left)
            cv2.imwrite(right_name, right)






if __name__ == '__main__':
    cti_path = "/opt/cvb-13.04.005/drivers/genicam/libGevTL.cti.1.2309"
    calib_path = "calibration/calib-11-01/calib.json"
    exposure_time = 25000
    save_dir = "/data"
    capture_homography(cti_path, calib_path, exposure_time, )
