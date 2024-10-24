import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from picamera2 import Picamera2
from libcamera import controls
import json

# Load camera parameters
mtx = np.load('camera/mtx.npy')  # Camera matrix
dist = np.load('camera/dist.npy')  # Distortion coefficients

# Load calibration data
with open('calibration_data.json', 'r') as f:
    calibration_data = json.load(f)

# Convert positions to numpy arrays
for marker_id in calibration_data:
    calibration_data[marker_id] = np.array(calibration_data[marker_id], dtype=np.float32)

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

try:
    while True:
        # Capture frame from camera
        frame = picam2.capture_array()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            ids = ids.flatten()

            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

            # Prepare lists for object points and image points
            object_points = []
            image_points = []

            for idx, marker_id in enumerate(ids):
                marker_id_str = str(marker_id)
                if marker_id_str in calibration_data:
                    # Known position of the marker in world coordinates
                    world_position = calibration_data[marker_id_str]

                    # Get the corner points of the marker
                    corner = corners[idx].reshape(-1, 2)
                    image_points.extend(corner)
                    
                    # Define the 3D coordinates of the marker's corners
                    half_size = marker_length / 2
                    obj_pts = np.array([
                        [-half_size, half_size, 0.0],
                        [half_size, half_size, 0.0],
                        [half_size, -half_size, 0.0],
                        [-half_size, -half_size, 0.0]
                    ], dtype=np.float32)

                    # Add the marker's world position to the object points
                    obj_pts += world_position
                    object_points.extend(obj_pts)
            
            if len(object_points) >= 4:
                object_points = np.array(object_points, dtype=np.float32)
                image_points = np.array(image_points, dtype=np.float32)

                # Estimate camera pose using solvePnP
                retval, rvec, tvec = cv2.solvePnP(object_points, image_points, mtx, dist)

                if retval:
                    # Draw the axes on the frame
                    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_length)

                    # Compute rotation matrix
                    R, _ = cv2.Rodrigues(rvec)

                    # Compute camera position
                    camera_position = -R.T @ tvec
                    camera_position = camera_position.flatten()
                    print(f"Camera Position: {camera_position}")

                    # Reset plot
                    ax.cla()
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    ax.set_zlim([-1, 1])

                    # Plot camera position
                    ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='r', label='Camera')

                    # Plot markers
                    for marker_id_str in calibration_data:
                        marker_pos = calibration_data[marker_id_str]
                        if int(marker_id_str) == 1:
                            ax.scatter(marker_pos[0], marker_pos[1], marker_pos[2], color='g', label='Marker 1')
                        else:
                            ax.scatter(marker_pos[0], marker_pos[1], marker_pos[2], color='b')

                    ax.legend()
                    plt.draw()
                    plt.pause(0.01)
                else:
                    print("Pose estimation failed.")
            else:
                print("Not enough points for pose estimation.")

            # Draw detected markers and axes
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_length / 2)
            cv2.aruco.drawDetectedMarkers(frame[0], corners, ids)
        else:
            print("No markers detected.")

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
