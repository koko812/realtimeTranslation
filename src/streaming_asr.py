import sounddevice as sd
import queue
import numpy as np
from scipy.io.wavfile import write
from google.cloud import speech
from datetime import datetime
import os
import threading

# === 設定 ===
samplerate = 16000
channels = 1
dtype = 'int16'
block_duration = 0.5  # 秒
save_dir = "streaming_logs"
os.makedirs(save_dir, exist_ok=True)

# === グローバル変数 ===
audio_queue = queue.Queue()
recorded_blocks = []
final_texts = []

# === 音声入力コールバック ===
def callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    audio_queue.put(bytes(indata))
    recorded_blocks.append(indata.copy())

# === 音声リクエスト生成 ===
def request_generator():
    while True:
        data = audio_queue.get()
        if data is None:
            break
        yield speech.StreamingRecognizeRequest(audio_content=data)

# === メイン処理 ===
def main():
    print("🎙 Enter で録音開始...")
    input()
    print("🎤 録音中（Enterで終了）")

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype=dtype, callback=callback):
        responses = client.streaming_recognize(config=streaming_config, requests=request_generator())
        def listen_print_loop():
            try:
                for response in responses:
                    for result in response.results:
                        transcript = result.alternatives[0].transcript
                        if result.is_final:
                            print("✅", transcript)
                            final_texts.append(transcript)
                        else:
                            print("🔄", transcript, end="\r")  # 暫定結果（上書き表示）
            except Exception as e:
                print("⚠️ 認識中にエラー:", e)



        # 非同期でレスポンスを表示
        print_thread = threading.Thread(target=listen_print_loop)
        print_thread.start()

        input()  # 録音停止の Enter
        audio_queue.put(None)
        print_thread.join()

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(save_dir, f"{timestamp}.wav")
    txt_path = os.path.join(save_dir, f"{timestamp}.txt")
    audio_np = np.concatenate(recorded_blocks, axis=0)
    write(wav_path, samplerate, audio_np)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_texts))
    print(f"\n💾 音声: {wav_path}")
    print(f"📝 認識結果: {txt_path}")

# === Google Cloud Speech-to-Text クライアント設定 ===
client = speech.SpeechClient()
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=samplerate,
    language_code="ja-JP",
)
streaming_config = speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True  # 暫定結果は表示しない
)

if __name__ == "__main__":
    main()

