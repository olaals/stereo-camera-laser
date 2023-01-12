import os
import rhovee as rv
from cv2 import aruco
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

def stereo_calibrate_cv2(left_stereo_paths, right_stereo_paths, lK, ldc, rK, rdc, board, verbose):
    # Load images
    limgs = [cv2.imread(path, 0) for path in left_stereo_paths]
    rimgs = [cv2.imread(path, 0) for path in right_stereo_paths]

    world_scaling = board.getSquareLength()
    rows = 6
    columns=5
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    objpoints = [] # 3d point in real world space
    img_points_left = []
    img_points_right = []


    for left_img, right_img in zip(limgs, rimgs):
        # Find the chess board corners aruco
        lcorners, lids, lchar_ids = aruco.detectMarkers(left_img, board.dictionary)
        rcorners, rids, rchar_ids = aruco.detectMarkers(right_img, board.dictionary)
        lret, lchar_corners, lchar_ids = aruco.interpolateCornersCharuco(lcorners, lids, left_img, board)
        rret, rchar_corners, rchar_ids = aruco.interpolateCornersCharuco(rcorners, rids, right_img, board)
        if lret and rret:
            all_corners_detected = len(lchar_corners) == rows*columns and len(rchar_corners) == rows*columns
        if lret and rret and all_corners_detected:
            objpoints.append(objp)
            img_points_left.append(lchar_corners)
            img_points_right.append(rchar_corners)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(objpoints, img_points_left, img_points_right, lK, ldc, rK, rdc, left_img.shape[::-1], flags=stereocalibration_flags)
    print(R)
    print(t)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3]= t.flatten()
    T = np.linalg.inv(T)
    return T, 0.1
    









def calib_left_right(calib_dir):
    left_calib_paths = glob.glob(os.path.join(calib_dir, 'cam-calib','left', '*.png'))
    right_calib_paths = glob.glob(os.path.join(calib_dir, 'cam-calib','right', '*.png'))
    left_stereo_paths = glob.glob(os.path.join(calib_dir, 'stereo-calib','left', '*.png'))
    right_stereo_paths = glob.glob(os.path.join(calib_dir, 'stereo-calib','right', '*.png'))
    left_stereo_paths.sort()
    right_stereo_paths.sort()
    left_calib_paths.sort()
    right_calib_paths.sort()
    cam_board = rv.cv.calib.load_charuco_board(os.path.join(calib_dir, 'cam-calib', 'board.json'))
    stereo_board = rv.cv.calib.load_charuco_board(os.path.join(calib_dir, 'stereo-calib', 'board.json'))

    print("Left calib paths", left_calib_paths)
    print("Right calib paths", right_calib_paths)

    lK, ldc, l_rms = rv.cv.calib.calibrate_camera(left_calib_paths, cam_board, req_markers=10, verbose=0)
    rK, rdc, r_rms = rv.cv.calib.calibrate_camera(right_calib_paths, cam_board, req_markers=10, verbose=0)
    print("Left cam mat", lK)
    print("Right cam mat", rK)
    print("Root mean square error for left camera: {}".format(l_rms))
    print("Root mean square error for right camera: {}".format(r_rms))
    #ltr, ltr_dev_angle= rv.cv.calib.calibrate_stereo_charuco(left_stereo_paths, right_stereo_paths, lK, ldc, rK, rdc, board, 1)
    ltr, ltr_dev_angle = stereo_calibrate_cv2(left_stereo_paths, right_stereo_paths, lK, ldc, rK, rdc, stereo_board, 1)


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
        "#debug_rms_left": l_rms,
        "#debug_rms_right": r_rms,
        "#debug_ltr_mean_dev_angle": ltr_dev_angle,
    }
    save_json(os.path.join(calib_dir, 'calib.json'), calib_dict)









if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stereo calibration')
    parser.add_argument('calib_dir', help='Calibration project dir')
    args = parser.parse_args()
    calib_left_right(args.calib_dir)    

    


