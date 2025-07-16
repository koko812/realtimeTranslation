import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import threading
import sys
import os
from google.cloud import speech
import tempfile

# Enter で録音開始して，もう一度押すと録音終了
# そして google asr api に送られて音声認識

samplerate = 16000
channels = 1
dtype = 'int16'

recording = []
recording_flag = False

def input_thread():
    global recording_flag
    input("🎙️ Enter を押すと録音開始します（もう一度押すと終了）: ")
    recording_flag = True
    print("🔴 録音中... Enter で停止")
    input()
    recording_flag = False
    print("⏹️ 録音停止")

def record_audio_dynamic():
    global recording, recording_flag
    block_duration = 0.5  # 秒単位

    def callback(indata, frames, time, status):
        if recording_flag:
            recording.append(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        dtype=dtype,
        callback=callback,
        blocksize=int(samplerate * block_duration),
    )

    with stream:
        input_thread()

    audio_np = np.concatenate(recording, axis=0)
    return audio_np

from datetime import datetime
import os

def transcribe_audio(array, samplerate, lang='ja-JP', save_dir="../recordings"):
    os.makedirs(save_dir, exist_ok=True)

    client = speech.SpeechClient()

    # タイムスタンプ名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(save_dir, f"{timestamp}.wav")
    txt_path = os.path.join(save_dir, f"{timestamp}.txt")

    # 録音を保存
    write(wav_path, samplerate, array)

    # 認識リクエスト
    with open(wav_path, "rb") as f:
        audio_data = f.read()

    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=samplerate,
        language_code=lang,
    )

    print("🧠 認識中...")
    response = client.recognize(config=config, audio=audio)

    transcripts = []
    if response.results:
        for result in response.results:
            t = result.alternatives[0].transcript
            print("📝 認識結果:", t)
            transcripts.append(t)
    else:
        print("⚠️ 認識結果が得られませんでした")

    # 結果をテキストで保存
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcripts))
    print(f"✅ 音声・テキストを保存しました: {wav_path}, {txt_path}")


if __name__ == "__main__":
    audio_data = record_audio_dynamic()
    transcribe_audio(audio_data, samplerate)

