import cv2
import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2
from libcamera import controls
import json
from scipy.spatial.transform import Rotation as R

# Load camera parameters
mtx = np.load('camera/mtx.npy')  # Camera matrix
dist = np.load('camera/dist.npy')  # Distortion coefficients

# Get ArUco marker dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

marker_length = 0.07  # meters
marker_count = 6

# Configure PiCamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Initialize storage for marker positions and orientations relative to marker ID 1
marker_poses = {}
for marker_id in range(1, marker_count + 1):  # IDs from 1 to 6
    if marker_id != 1:
        marker_poses[marker_id] = {'rotations': [], 'positions': []}

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

                # For other markers, compute their positions and orientations relative to marker 1
                for idx, marker_id in enumerate(ids):
                    if marker_id != 1 and marker_id <= marker_count:
                        rvec, tvec = rvecs[idx], tvecs[idx]
                        R_marker, _ = cv2.Rodrigues(rvec)
                        T_marker = tvec.reshape((3, 1))
                        RT_marker = np.hstack((R_marker, T_marker))
                        RT_marker = np.vstack((RT_marker, [0, 0, 0, 1]))

                        # Compute transformation from marker 1 to this marker
                        RT_relative = np.dot(np.linalg.inv(RT_1), RT_marker)

                        # Extract rotation and translation
                        R_relative = RT_relative[:3, :3]
                        T_relative = RT_relative[:3, 3]

                        # Convert rotation matrix to quaternion
                        rotation = R.from_matrix(R_relative).as_quat()

                        # Store rotation and position
                        marker_poses[marker_id]['rotations'].append(rotation)
                        marker_poses[marker_id]['positions'].append(T_relative)

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

# Compute average positions and rotations
for marker_id in marker_poses:
    data = marker_poses[marker_id]
    if len(data['positions']) > 0:
        # Average positions
        avg_position = np.mean(data['positions'], axis=0)

        # Average rotations using quaternions
        rotations = R.from_quat(data['rotations'])
        mean_rotation = rotations.mean()
        avg_rotation_matrix = mean_rotation.as_matrix()

        # Store the averaged pose
        marker_poses[marker_id]['average_position'] = avg_position
        marker_poses[marker_id]['average_rotation'] = avg_rotation_matrix

        print(f"Marker ID {marker_id} average position relative to Marker ID 1: {avg_position}")
        print(f"Marker ID {marker_id} average rotation matrix relative to Marker ID 1:\n{avg_rotation_matrix}")
    else:
        print(f"No data collected for Marker ID {marker_id}")

# Save the calibration data
calibration_data = {}
for marker_id, data in marker_poses.items():
    if 'average_position' in data and 'average_rotation' in data:
        calibration_data[marker_id] = {
            'position': data['average_position'].tolist(),
            'rotation_matrix': data['average_rotation'].tolist()
        }

with open('calibration_data.json', 'w') as f:
    json.dump(calibration_data, f)

print("Calibration data saved to 'calibration_data.json'")
