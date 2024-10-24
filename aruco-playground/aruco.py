import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from picamera2 import Picamera2
from libcamera import controls

# カメラパラメータの読み込み
mtx = np.load('camera/mtx.npy')  # カメラ行列
dist = np.load('camera/dist.npy')  # 歪み係数

# ArUcoマーカーディクショナリの取得
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# マーカーの物理的なサイズ (メートル)
marker_length = 0.07

# PiCameraの設定
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# 3Dプロットのセットアップ
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 3Dグラフの範囲設定
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

while True:
    # カメラからフレームを取得
    frame = picam2.capture_array()

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ArUcoマーカーの検出
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # マーカーが検出された場合
    if np.all(ids is not None):
        # マーカーの姿勢（回転ベクトルと並進ベクトル）を推定
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

        # 各マーカーについて位置を推定
        for rvec, tvec in zip(rvecs, tvecs):
            # マーカーの座標軸を描画
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_length / 2)

            # 回転ベクトルを回転行列に変換
            R, _ = cv2.Rodrigues(rvec)

            # カメラ座標を計算
            R_T = R.T  # 回転行列の転置
            camera_position = -np.dot(R_T, tvec.T).squeeze()

            # プロットをリセット
            ax.cla()  # 以前のプロットをクリア
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim([-1, 1])  # 軸の範囲を再設定
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            # カメラの位置を3Dプロットに反映
            ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='b')
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      R_T[0, 0], R_T[0, 1], R_T[0, 2], length=0.1, color='r')
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      R_T[1, 0], R_T[1, 1], R_T[1, 2], length=0.1, color='g')
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                      R_T[2, 0], R_T[2, 1], R_T[2, 2], length=0.1, color='b')

    # 検出されたマーカーの枠を表示
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # 結果を表示
    cv2.imshow('frame', frame)

    # 3Dグラフを更新
    plt.draw()
    plt.pause(0.01)

    # 'Esc'キーが押されたら終了
    if cv2.waitKey(1) == 27:
        break

# 後片付け
picam2.stop()
cv2.destroyAllWindows()
plt.close()
