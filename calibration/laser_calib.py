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

def average_distance_point_to_plane(u, pts):
    d = np.abs(np.dot(pts, u[:3]) + u[-1])
    return np.mean(d)

def get_proj_planar_homography(T12, u1, K1, K2):
    print("T12", T12)
    d = u1[-1]
    n = u1[:3]
    T_21 = np.linalg.inv(T12)
    R_21 = T_21[:3,:3]
    t_21 = T_21[:3,3]
    min_term = (1/d)*np.outer(t_21,n)
    print("min term shape", min_term.shape)
    H21 = R_21 - min_term
    H21 = K2@H21@np.linalg.inv(K1)
    H12 = np.linalg.inv(H21)

    return H12

def calib_stereo_laser(calib_dir, verbose=1):
    THRESH = 150
    calib_json_path = os.path.join(calib_dir, 'calib.json')
    with open(calib_json_path, 'r') as f:
        calib_dict = json.load(f)
    T_ltr, lK, ldc, rK, rdc = read_calib_json(calib_dict)
    print("T_ltr", T_ltr.shape)

    left_laser_img_paths = glob.glob(os.path.join(calib_dir, 'laser-calib','left', '*.png'))
    right_laser_img_paths = glob.glob(os.path.join(calib_dir, 'laser-calib','right', '*.png'))
    left_laser_img_paths.sort()
    right_laser_img_paths.sort()
    print('Found {} left laser images and {} right laser images'.format(len(left_laser_img_paths), len(right_laser_img_paths)))
    all_pts = []
    for left_img_path, right_img_path in zip(left_laser_img_paths, right_laser_img_paths):
        left_img = cv2.imread(left_img_path, 0)
        right_img = cv2.imread(right_img_path, 0)
        laser_pts = rv.cv.laser.triangulate_laser_lines(left_img, right_img, THRESH, T_ltr[:3,:3], T_ltr[:3,-1], lK, ldc, rK, rdc, verbose=verbose)
        if len(laser_pts) > 0:
            all_pts.append(laser_pts)

    # vertical stack of all_pts
    all_pts = np.vstack(all_pts)
    u = rv.cv.calib.fit_plane_svd(all_pts)
    avg_plane_error = average_distance_point_to_plane(u, all_pts)
    H = get_proj_planar_homography(T_ltr, u, lK, rK)
    print("H" , H)
    print("u", u)

    calib_dict['H'] = H.tolist()
    calib_dict['u'] = u.tolist()
    calib_dict['#debug_laser_plane_error'] = avg_plane_error
    save_json(os.path.join(calib_dir, 'calib.json'), calib_dict)


    # plot 3d points
    if verbose > -1:
        print("Avg plane error", avg_plane_error)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), all_pts[:,1].max()-all_pts[:,1].min(), all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
        mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
        mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
        mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        # scatter small points
        ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2], s=0.01, c='r', marker='o')
        #ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2], c='r', marker='o')
        # draw plane on top of points
        x = np.linspace(all_pts[:,0].min(), all_pts[:,0].max(), 100)
        y = np.linspace(all_pts[:,1].min(), all_pts[:,1].max(), 100)
        X, Y = np.meshgrid(x, y)
        Z = (-u[0]*X - u[1]*Y - u[-1]) / u[2]
        ax.plot_surface(X, Y, Z, alpha=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()



        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stereo laser calibration')
    parser.add_argument('calib_dir', help='Calibration project dir')
    parser.add_argument('--verbose', help='Calibration project dir', default=0, type=int)
    args = parser.parse_args()
    calib_stereo_laser(args.calib_dir, args.verbose)
    


