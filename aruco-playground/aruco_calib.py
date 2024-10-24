import cv2
import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2
from libcamera import controls
import json

# Load camera parameters
mtx = np.load('camera/mtx.npy')  # Camera matrix
dist = np.load('camera/dist.npy')  # Distortion coefficients

# Get ArUco marker dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

marker_length = 0.07  # meters

# Configure PiCamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Initialize storage for marker positions relative to marker ID 1
marker_positions = {}
for marker_id in range(1, 7):  # IDs from 1 to 6
    if marker_id != 1:
        marker_positions[marker_id] = {'position': np.zeros((3,)), 'count': 0}

try:
    while True:
        # Capture frame from camera
        frame = picam2.capture_array()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            ids = ids.flatten()
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

            if 1 in ids:
                # Get index of marker ID 1
                idx_1 = np.where(ids == 1)[0][0]
                rvec_1, tvec_1 = rvecs[idx_1], tvecs[idx_1]

                # Transformation matrix from marker 1 to camera
                R_1, _ = cv2.Rodrigues(rvec_1)
                T_1 = tvec_1.reshape((3, 1))
                RT_1 = np.hstack((R_1, T_1))
                RT_1 = np.vstack((RT_1, [0, 0, 0, 1]))

                # For other markers, compute their positions relative to marker 1
                for idx, marker_id in enumerate(ids):
                    if marker_id != 1:
                        rvec, tvec = rvecs[idx], tvecs[idx]
                        R, _ = cv2.Rodrigues(rvec)
                        T = tvec.reshape((3, 1))
                        RT = np.hstack((R, T))
                        RT = np.vstack((RT, [0, 0, 0, 1]))

                        # Compute transformation from marker 1 to this marker
                        RT_relative = np.dot(np.linalg.inv(RT_1), RT)

                        # Extract translation vector
                        position = RT_relative[:3, 3]

                        # Accumulate position for averaging
                        marker_positions[marker_id]['position'] += position
                        marker_positions[marker_id]['count'] += 1

            # Draw detected markers and axes
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_length / 2)
            cv2.aruco.drawDetectedMarkers(frame[0], corners, ids)

        # Display frame
        cv2.imshow('Calibration', frame)

        # Break loop on 'Esc' key press
        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    pass

# Stop camera and close windows
picam2.stop()
cv2.destroyAllWindows()

# Compute average positions
for marker_id in marker_positions:
    data = marker_positions[marker_id]
    if data['count'] > 0:
        average_position = data['position'] / data['count']
        marker_positions[marker_id]['average_position'] = average_position
        print(f"Marker ID {marker_id} average position relative to Marker ID 1: {average_position}")
    else:
        print(f"No data collected for Marker ID {marker_id}")

# Save the calibration data
calibration_data = {marker_id: data['average_position'].tolist() for marker_id, data in marker_positions.items() if 'average_position' in data}

with open('calibration_data.json', 'w') as f:
    json.dump(calibration_data, f)

print("Calibration data saved to 'calibration_data.json'")
