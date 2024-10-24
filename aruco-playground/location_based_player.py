import socket
import json
import pygame
import numpy as np
import soundfile as sf

# サーバーの設定
host = '127.0.0.1'
port = 65432

# 音声ファイルを読み込む
file1 = 'berlin1936.wav'
file2 = 'warsaw1939.wav'

data1, samplerate1 = sf.read(file1)
data2, samplerate2 = sf.read(file2)

# 音声データがステレオかモノラルかチェック
if len(data1.shape) == 1:
    data1 = np.tile(data1, (2, 1)).T  # モノラルの場合、ステレオに変換
if len(data2.shape) == 1:
    data2 = np.tile(data2, (2, 1)).T  # モノラルの場合、ステレオに変換

# pygame の初期化
pygame.mixer.init(frequency=samplerate1, size=-16, channels=2)

# WAVデータの長さを確認し、長さをそろえる
min_length = min(len(data1), len(data2))
data1 = data1[:min_length]
data2 = data2[:min_length]

# サーバーを設定
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.bind((host, port))
server_sock.listen()

print('Waiting for connection...')
conn, addr = server_sock.accept()
print(f"Connected by {addr}")

def get_volume_from_position(x):
    """
    x座標に基づいて音量を計算する。
    x < -0.2: 左（file1の音だけ聞こえる）
    -0.2 <= x <= 0.2: file1とfile2の音量を線形に変化
    x > 0.2: 右（file2の音だけ聞こえる）
    """
    if x <= -0.2:
        return 1.0, 0.0  # 完全に左寄り -> file1が最大音量、file2は無音
    elif x >= 0.2:
        return 0.0, 1.0  # 完全に右寄り -> file2が最大音量、file1は無音
    else:
        left_volume = (0.2 - x) / 1.0  # 中央に近いと両方の音が聞こえる
        right_volume = (x + 0.2) / 1.0
        return left_volume, right_volume

try:
    while True:
        data = conn.recv(1024)
        if not data:
            break
        position_data = json.loads(data.decode('utf-8'))
        x_pos = position_data['position'][0]

        # x座標に基づいて音量を調整
        left_vol, right_vol = get_volume_from_position(x_pos)

        # 左右の音量を適用してミキシング
        mixed_data = (data1 * left_vol + data2 * right_vol).astype(np.float32)

        # 音声を再生
        sound = pygame.mixer.Sound(mixed_data)
        sound.play()

        # 再生中に次のデータを受け取る
        while pygame.mixer.get_busy():
            pass

except KeyboardInterrupt:
    print("Server stopped.")

finally:
    conn.close()
    server_sock.close()
    pygame.mixer.quit()
