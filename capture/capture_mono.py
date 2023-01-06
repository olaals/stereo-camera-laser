import sys
sys.path.append('../')
import cv2
import os
from genicam_wrappers import GenicamMono



def capture(save_dir, cti_path, device_num, exposure_time, gray=True):
    cap = GenicamMono(cti_path, device_num, exposure_time, gray=True)
    save_num = 0
    os.makedirs(save_dir, exist_ok=True)
    while True:
        img = cap.read()
        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            save_num += 1
            cv2.imwrite(save_dir+f'/img{format(save_num, "02d")}.png', img)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='data')
    parser.add_argument('--device_num', type=int)
    parser.add_argument('--exp_time', type=int, default=60000)
    args = parser.parse_args()

    cti_path = "/opt/cvb-13.04.001/drivers/genicam/libGevTL.cti.1.2305"
    exposure_time = args.exp_time
    #capture_stereo(args.save_dir, cti_path, device_num_left, device_num_right, exposure_time)
    capture(args.save_dir, cti_path, args.device_num, args.exp_time)
