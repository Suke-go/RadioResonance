import cv2
import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2
from libcamera import controls
import json
from scipy.spatial.transform import Rotation as R

# Load calibration data
with open('calibration_data.json', 'r') as f:
    calibration_data = json.load(f)

# Convert marker positions and rotations to numpy arrays
for marker_id in calibration_data:
    data = calibration_data[marker_id]
    calibration_data[marker_id]['position'] = np.array(data['position'])
    calibration_data[marker_id]['rotation_matrix'] = np.array(data['rotation_matrix'])

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

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Plot marker positions
for marker_id in calibration_data:
    position = calibration_data[marker_id]['position']
    ax.scatter(position[0], position[1], position[2], color='red', marker='^')
    ax.text(position[0], position[1], position[2], f"Marker {marker_id}", color='red')

try:
    while True:
        # Capture frame from camera
        frame = picam2.capture_array()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        camera_positions = []
        camera_orientations = []

        if ids is not None:
            ids = ids.flatten()
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

            for idx, marker_id in enumerate(ids):
                marker_id_str = str(marker_id)
                if marker_id_str in calibration_data:
                    # Get pose of the marker relative to the camera
                    rvec, tvec = rvecs[idx], tvecs[idx]

                    # Convert rotation vector to rotation matrix
                    R_ct, _ = cv2.Rodrigues(rvec)
                    T_ct = tvec.reshape((3, 1))

                    # Transformation matrix from camera to marker (camera coordinate system to marker coordinate system)
                    RT_cam_to_marker = np.hstack((R_ct, T_ct))
                    RT_cam_to_marker = np.vstack((RT_cam_to_marker, [0, 0, 0, 1]))

                    # Inverse to get marker to camera transformation
                    RT_marker_to_cam = np.linalg.inv(RT_cam_to_marker)

                    # Known marker pose in world coordinate system
                    marker_position = calibration_data[marker_id_str]['position']
                    marker_rotation = calibration_data[marker_id_str]['rotation_matrix']

                    # Transformation matrix from world to marker
                    RT_world_to_marker = np.hstack((marker_rotation, marker_position.reshape((3, 1))))
                    RT_world_to_marker = np.vstack((RT_world_to_marker, [0, 0, 0, 1]))

                    # Compute camera pose in world coordinate system
                    RT_world_to_cam = np.dot(RT_world_to_marker, RT_marker_to_cam)

                    # Extract rotation and translation
                    R_wc = RT_world_to_cam[:3, :3]
                    T_wc = RT_world_to_cam[:3, 3]

                    camera_positions.append(T_wc)
                    camera_orientations.append(R_wc)

                    # Draw detected markers and axes
                    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_length / 2)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if camera_positions:
                # Average camera positions and orientations if multiple markers are detected
                avg_camera_position = np.mean(camera_positions, axis=0)
                avg_camera_orientation = R.from_matrix(camera_orientations).mean().as_matrix()

                # Reset plot
                ax.cla()
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_zlabel("Z (m)")
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])

                # Plot marker positions
                for marker_id in calibration_data:
                    position = calibration_data[marker_id]['position']
                    ax.scatter(position[0], position[1], position[2], color='red', marker='^')
                    ax.text(position[0], position[1], position[2], f"Marker {marker_id}", color='red')

                # Plot camera position
                ax.scatter(avg_camera_position[0], avg_camera_position[1], avg_camera_position[2], color='blue', marker='o')
                ax.text(avg_camera_position[0], avg_camera_position[1], avg_camera_position[2], "Camera", color='blue')

                # Draw camera orientation axes
                axis_length = 0.05  # Adjust as needed
                origin = avg_camera_position
                x_axis = avg_camera_orientation[:, 0] * axis_length
                y_axis = avg_camera_orientation[:, 1] * axis_length
                z_axis = avg_camera_orientation[:, 2] * axis_length

                ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r')
                ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g')
                ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b')

                plt.draw()
                plt.pause(0.001)

        # Display frame
        cv2.imshow('Localization', frame)

        # Break loop on 'Esc' key press
        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    pass

# Stop camera and close windows
picam2.stop()
cv2.destroyAllWindows()
plt.close()
