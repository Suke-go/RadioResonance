import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from picamera2 import Picamera2
from libcamera import controls

print("Loading camera parameters...")
mtx = np.load('camera/mtx.npy')  # カメラ行列
dist = np.load('camera/dist.npy')  # 歪み係数
print("Camera parameters loaded.")

print("Getting ArUco marker dictionary...")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()
print("ArUco marker dictionary obtained.")

marker_length = 0.07
print(f"Marker length set to {marker_length} meters.")

print("Configuring PiCamera...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
print("PiCamera configured and started.")

print("Setting up 3D plot...")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
print("3D plot setup complete.")

while True:
    print("Capturing frame from camera...")
    frame = picam2.capture_array()
    print("Frame captured.")

    print("Converting frame to grayscale...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Conversion to grayscale complete.")

    print("Detecting ArUco markers...")
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(f"Markers detected: {ids}")

    if np.all(ids is not None):
        print("Estimating pose of detected markers...")
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)
        print("Pose estimation complete.")

        for rvec, tvec in zip(rvecs, tvecs):
            print("Drawing frame axes for marker...")
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_length / 2)
            print("Frame axes drawn.")

            print("Converting rotation vector to rotation matrix...")
            R, _ = cv2.Rodrigues(rvec)
            print("Conversion complete.")

            print("Calculating camera position...")
            R_T = R.T
            camera_position = -np.dot(R_T, tvec.T).squeeze()
            print(f"Camera position: {camera_position}")

            print("Resetting plot...")
            ax.cla()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            print("Plot reset.")

            print("Updating 3D plot with camera position...")
            ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='b')
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      R_T[0, 0], R_T[0, 1], R_T[0, 2], length=0.1, color='r')
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      R_T[1, 0], R_T[1, 1], R_T[1, 2], length=0.1, color='g')
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      R_T[2, 0], R_T[2, 1], R_T[2, 2], length=0.1, color='b')
            print("3D plot updated.")

    if corners:
        print("Drawing detected markers on frame...")
        # Remove the alpha channel from the frame to convert it to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        print("Detected markers drawn.")

        print("Displaying frame...")
        cv2.imshow('frame', frame)
        print("Frame displayed.")

        print("Updating 3D plot...")
        plt.draw()
        plt.pause(0.01)
        print("3D plot updated.")

    if cv2.waitKey(1) == 27:
        print("Escape key pressed. Exiting loop.")
        break

print("Stopping camera...")
picam2.stop()
print("Camera stopped.")

print("Destroying all windows...")
cv2.destroyAllWindows()
plt.close()
print("All windows destroyed.")
