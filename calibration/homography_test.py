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


def combine_4_images_to_2by2grid(img1, img2, img3, img4):
    # if gray convert to rgb and resize to 256x256
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    if len(img3.shape) == 2:
        img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2RGB)
    if len(img4.shape) == 2:
        img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2RGB)

    # combine images to 2 by 2 grid with image size 512x512
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    img3 = cv2.resize(img3, (256, 256))
    img4 = cv2.resize(img4, (256, 256))
    img1 = cv2.copyMakeBorder(img1, 0, 256, 0, 256, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img2 = cv2.copyMakeBorder(img2, 0, 256, 256, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img3 = cv2.copyMakeBorder(img3, 256, 0, 0, 256, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img4 = cv2.copyMakeBorder(img4, 256, 0, 256, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.vconcat([img1, img3])
    img = cv2.hconcat([img, img2])
    img = cv2.hconcat([img, img4])
    return img





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

        # 2x2 matplotlib grid
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(undist_left, cmap='gray')
        axs[0, 0].set_title('Left')
        axs[0, 1].imshow(undist_right, cmap='gray')
        axs[0, 1].set_title('Right')
        axs[1, 0].imshow(projected, cmap='gray')
        axs[1, 0].set_title('Projected')
        axs[1, 1].imshow(depth_stack)
        axs[1, 1].set_title('Depth Stack')
        plt.show()









        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stereo laser calibration')
    parser.add_argument('calib_dir', help='Calibration project dir')
    args = parser.parse_args()
    test_homography(args.calib_dir)
    


