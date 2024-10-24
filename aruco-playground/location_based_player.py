import socket
import json
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa

# 音声ファイルの読み込み
audio1 = AudioSegment.from_file("berlin1936.wav")  # 音源1
audio2 = AudioSegment.from_file("warsaw1939.wav")  # 音源2

# ソケットの設定
host = '127.0.0.1'
port = 65432

# サーバーソケットの作成
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.bind((host, port))
server_sock.listen()

print("接続を待機中...")
conn, addr = server_sock.accept()
print(f"接続されました: {addr}")

# 音を再生するための関数
def play_audio(audio1, audio2, volume1, volume2):
    # ボリューム調整
    adjusted_audio1 = audio1 + volume1
    adjusted_audio2 = audio2 + volume2
    
    # 同時再生のため、長さを揃える
    combined_audio = adjusted_audio1.overlay(adjusted_audio2)
    
    # 再生 (blocking call: 再生完了まで次の処理は進まない)
    play_obj = sa.play_buffer(
        combined_audio.raw_data,
        num_channels=combined_audio.channels,
        bytes_per_sample=combined_audio.sample_width,
        sample_rate=combined_audio.frame_rate
    )
    play_obj.wait_done()  # 再生が終わるまで待つ

# クライアントからのデータ受信と処理
try:
    while True:
        data = conn.recv(1024)
        if not data:
            break

        # 受信したデータをデコードしてJSONとしてパース
        received_data = json.loads(data.decode('utf-8'))
        position = received_data["position"]
        x = position[0]  # X座標を取得

        # X座標に応じたボリューム調整
        if x <= -0.5:
            volume1 = 0  # 左に行き過ぎたら音源1のみ聞こえる
            volume2 = -100  # 音源2の音量を最小に
        elif x >= 0.5:
            volume1 = -100  # 右に行き過ぎたら音源2のみ聞こえる
            volume2 = 0  # 音源1の音量を最小に
        else:
            # xが-0.5から0.5の間にある場合、音量を線形補間で調整
            volume1 = (0.5 - x) * 200 - 100  # 0.5で音源1が最大、-0.5で最小
            volume2 = (x + 0.5) * 200 - 100  # -0.5で音源2が最大、0.5で最小

        # 調整後の音を再生
        play_audio(audio1, audio2, volume1, volume2)

except KeyboardInterrupt:
    pass
finally:
    # ソケットを閉じる
    conn.close()
    server_sock.close