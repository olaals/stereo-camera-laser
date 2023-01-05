import os
import rhovee as rv
import numpy as np
import cv2
import glob

def R_to_axis_angle(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    axis = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(angle))
    return axis, angle

def save_json(filename, data):
    import json
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def calib_left_right(calib_dir):
    left_paths = glob.glob(os.path.join(calib_dir, 'board-imgs','left', '*.png'))
    right_paths = glob.glob(os.path.join(calib_dir, 'board-imgs','right', '*.png'))
    print('Found {} left images and {} right images'.format(len(left_paths), len(right_paths)))
    board = rv.cv.calib.load_charuco_board(os.path.join(calib_dir, 'board.json'))
    ltr, lK, ldc, rK, rdc = rv.cv.calib.calibrate_stereo_charuco(left_paths, right_paths, board)
    print('Left camera matrix: {}'.format(lK))
    print('Left distortion: {}'.format(ldc))
    print('Right camera matrix: {}'.format(rK))
    print('Right distortion: {}'.format(rdc))
    print('Left to right transform: {}'.format(ltr))
    axis, angle = R_to_axis_angle(ltr[:3,:3])
    angle = angle * 180 / np.pi
    print('Left to right rotation axis: {}'.format(axis))
    print('Left to right rotation angle: {}'.format(angle))
    calib_dict = {
        'left_K': lK.tolist(),
        'left_dist': ldc.tolist(),
        'right_K': rK.tolist(),
        'right_dist': rdc.tolist(),
        'T_ltr': ltr.tolist(),
    }
    save_json(os.path.join(calib_dir, 'calib.json'), calib_dict)









if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stereo calibration')
    parser.add_argument('calib_dir', help='Calibration project dir')
    args = parser.parse_args()
    calib_left_right(args.calib_dir)    

    


