import cv2
import numpy as np
import os

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

# ウェブカメラの起動
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcamの起動に失敗しました")
    exit()

print("キャリブレーション用のチェスボードをウェブカメラに向けてください。'q'を押して終了します。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました")
        break

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
    cv2.imshow('Webcam', gray)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャリブレーションの実行
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # キャリブレーション結果を表示
    print('カメラ行列 (mtx): \n', mtx)
    print('歪み係数 (dist): \n', dist)

    # 結果を保存
    np.save(os.path.join(save_dir, 'mtx.npy'), mtx)
    np.save(os.path.join(save_dir, 'dist.npy'), dist)

    print(f"カメラ行列と歪み係数を{save_dir}に保存しました")
else:
    print("キャリブレーションに失敗しました。十分なコーナーが検出されませんでした。")

# ウェブカメラをリリース
cap.release()
cv2.destroyAllWindows()
