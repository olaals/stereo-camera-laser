import sys
sys.path.append('../')
import cv2
import os
from genicam_wrappers import GenicamStereo


def capture_stereo(save_dir, cti_path, dn_left, dn_right, exposure_time, gray=True):
    cap = GenicamStereo(cti_path, dn_left, dn_right, exposure_time, gray=True)
    save_num = 0
    left_save_dir = save_dir + '/left/'
    right_save_dir = save_dir + '/right/'
    os.makedirs(left_save_dir, exist_ok=True)
    os.makedirs(right_save_dir, exist_ok=True)
    while True:
        left, right = cap.read()
        cv2.imshow('left', left)
        cv2.imshow('right', right)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            save_num += 1
            cv2.imwrite(left_save_dir+f'img{format(save_num, "02d")}.png', left)
            cv2.imwrite(right_save_dir+f'img{format(save_num, "02d")}.png', right)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='data')
    args = parser.parse_args()

    cti_path = "/opt/cvb-13.04.001/drivers/genicam/libGevTL.cti.1.2305"
    exposure_time = 60000
    device_num_left = 1
    device_num_right = 0
    capture_stereo(args.save_dir, cti_path, device_num_left, device_num_right, exposure_time)
