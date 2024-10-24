import cv2
import numpy as np
import os
from picamera2 import Picamera2
from libcamera import controls

# キャリブレーションに使用するチェスボードのサイズ（コーナーの数）
chessboard_size = (9, 6)

# キャリブレーションパラメータの保存ディレクトリ
save_dir = './camera'

# 保存ディレクトリが存在しない場合は作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# チェスボードの3D座標（Zは0、平面）
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 3Dおよび2D座標を格納する配列
objpoints = []  # 3D実世界空間の座標
imgpoints = []  # 2D画像平面上の座標

# 終了基準（コーナー検出の精度を調整）
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# PiCameraの起動
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

print("キャリブレーション用のチェスボードをカメラに向けてください。'Esc'を押して終了します。")

while True:
    frame = picam2.capture_array()

    # グレースケール画像に変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # チェスボードのコーナーを検出
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        print("コーナーが検出されました")
        # コーナーを精緻化
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)

        # 検出したコーナーを画像に描画
        frame = cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
    else:
        print("コーナーが検出されませんでした")

    # 画像を表示
    cv2.imshow('Camera', gray)

    # 'Esc'キーで終了
    if cv2.waitKey(1) == 27:
        break

print("Stopping the camera...")
picam2.stop()
print("Camera stopped.")

print("Starting camera calibration...")
if len(objpoints) > 0 and len(imgpoints) > 0:
    print(f"Number of object points: {len(objpoints)}")
    print(f"Number of image points: {len(imgpoints)}")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration completed.")
    print(f"Calibration success: {ret}")

    # キャリブレーション結果を表示
    print('カメラ行列 (mtx): \n', mtx)
    print('歪み係数 (dist): \n', dist)

    # 結果を保存
    print("Saving calibration results...")
    np.save(os.path.join(save_dir, 'mtx.npy'), mtx)
    np.save(os.path.join(save_dir, 'dist.npy'), dist)
    print(f"カメラ行列と歪み係数を{save_dir}に保存しました")
else:
    print("キャリブレーションに失敗しました。十分なコーナーが検出されませんでした。")
    print(f"Number of object points: {len(objpoints)}")
    print(f"Number of image points: {len(imgpoints)}")

# カメラを停止
print("Destroying all windows...")
cv2.destroyAllWindows()
print("All windows destroyed.")
