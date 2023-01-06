import os
import rhovee as rv
import numpy as np
import cv2
import glob
import json
import matplotlib.pyplot as plt
from rhovee.cv import calib

def R_to_axis_angle(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    axis = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(angle))
    return axis, angle

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def read_calib_json(calib_dict):
    lK = np.array(calib_dict['left_K'])
    ldc = np.array(calib_dict['left_dist'])
    rK = np.array(calib_dict['right_K'])
    rdc = np.array(calib_dict['right_dist'])
    T_ltr = np.array(calib_dict['T_ltr'])
    return T_ltr, lK, ldc, rK, rdc


def apply_planar_homography(H, img):
    projected_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    return projected_img


def test_homography(calib_dir):
    calib_json_path = os.path.join(calib_dir, 'calib.json')
    with open(calib_json_path, 'r') as f:
        calib_dict = json.load(f)
    T_ltr, lK, ldc, rK, rdc = read_calib_json(calib_dict)
    H = np.array(calib_dict['H'])
    print("H", H)
    print("T_ltr", T_ltr)
    #H = np.linalg.inv(H)
    u = np.array(calib_dict['u'])
    print("T_ltr", T_ltr.shape)

    left_laser_img_paths = glob.glob(os.path.join(calib_dir, 'laser-calib','left', '*.png'))
    right_laser_img_paths = glob.glob(os.path.join(calib_dir, 'laser-calib','right', '*.png'))
    print('Found {} left laser images and {} right laser images'.format(len(left_laser_img_paths), len(right_laser_img_paths)))
    for left_img_path, right_img_path in zip(left_laser_img_paths, right_laser_img_paths):
        left_img = cv2.imread(left_img_path, 0)
        right_img = cv2.imread(right_img_path, 0)
        undist_right = cv2.undistort(right_img, rK, rdc)
        undist_left = cv2.undistort(left_img, lK, ldc)
        projected = apply_planar_homography(H, undist_right)
        # depth stack images to rgb
        depth_stack = np.dstack((undist_left, projected, projected))
        cv2.imshow('depth stack', depth_stack)
        cv2.waitKey(0)





        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stereo laser calibration')
    parser.add_argument('calib_dir', help='Calibration project dir')
    args = parser.parse_args()
    test_homography(args.calib_dir)
    


