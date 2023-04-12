# Stereo camera laser

Scripts for calibrating and capturing images with a stereo laser scanner. 
The calibration gives the camera matrix and distortion coefficients of each camera,
as well as the rotation and translation between the cameras, and the laser plane.

# Calibrating each camera individually
## Print charuco board
To get the camera matrix of each camera, a charuco board must be printed. 
Use [create_charuco_board.py](calibration/create_charuco_board.py) with the following arguments
```bash
python create_charuco_board.py 
	--save_dir calib_dir 
	--aruco_dict_str DICT_APRILTAG_16H5  # See [OpenCV charuco dicts](https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#ggac84398a9ed9dd01306592dd616c2c975a6eb1a3e9c94c7123d8b1904a57193f16)
	--sq_x 7 # number of horizontal squares
```

