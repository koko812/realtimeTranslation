
import sounddevice as sd
import queue
import numpy as np
import wave
import uuid
import json
import os
import threading
from datetime import datetime
from html import unescape
from google.cloud import speech
from google.cloud import translate_v2 as translate
import sys


def print_inline(text):
    sys.stdout.write("\033[2K\r" + text + "\n")  # カーソル行をクリアして上書き
    sys.stdout.flush()


def print_interim_block(text, translation):
    sys.stdout.write("\033[2A")  # カーソルを2行上へ
    sys.stdout.write("\033[2K\r")  # 1行目消去
    sys.stdout.write(f"🔄 {text}\n")
    sys.stdout.write("\033[2K\r")  # 2行目消去
    sys.stdout.write(f"🌍 {translation}\n")
    sys.stdout.flush()


# === 設定 ===
SAMPLE_RATE = 16000
CHANNELS = 1
LANGUAGE_CODE = "ja-JP"
LOG_DIR = "../logs"
AUDIO_DIR = "../audio"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# === セッションロガー ===
class SessionLogger:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self.start_time = datetime.now().isoformat()
        self.audio_bytes = 0
        self.chunks = 0
        self.results = []
        self.audio_raw = []

    def log_chunk(self, data):
        self.audio_bytes += len(data)
        self.chunks += 1
        self.audio_raw.append(data)

    def log_result(self, text, confidence, is_final, translation=None):
        self.results.append({
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "translation": translation,
            "confidence": confidence,
            "final": is_final
        })


    def save(self, wav_path):
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16 = 2bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(self.audio_raw))
        log = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "language_code": LANGUAGE_CODE,
            "audio_bytes": self.audio_bytes,
            "chunks": self.chunks,
            "audio_file": wav_path,
            "results": self.results
        }
        with open(f"{LOG_DIR}/{self.session_id}.json", "w") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)


translate_client = translate.Client()
def translate_text(text, target_lang="zh"):
    result = translate_client.translate(text, target_language=target_lang)
    return unescape(result['translatedText'])

# === 音声録音 ===
def record_audio(audio_queue, stop_event):
    def callback(indata, frames, time, status):
        if not stop_event.is_set():
            audio_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=callback):
        print("🎤 録音中（Enterで終了）")
        input()
        stop_event.set()
        print("🛑 録音終了")

# === ストリーミング音声認識 ===
def streaming_asr():
    print("🎙 Enterで録音開始...")
    input()

    audio_queue = queue.Queue()
    stop_event = threading.Event()
    logger = SessionLogger()
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE_CODE,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    def request_gen():
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.5)
                logger.log_chunk(data)
                yield speech.StreamingRecognizeRequest(audio_content=data)
            except queue.Empty:
                continue

    recorder_thread = threading.Thread(target=record_audio, args=(audio_queue, stop_event))
    recorder_thread.start()

    # 最初に空行を確保（interim 上書きスペース）
    print("\n\n", end="")
    last_interim_text = ""
    last_interim_translation = ""

    try:
        responses = client.streaming_recognize(config=streaming_config, requests=request_gen())
        for response in responses:
            if not response.results:
                continue

            # 全ての result の transcript を結合
            full_text = " ".join(r.alternatives[0].transcript.replace("\n", " ") for r in response.results)
            last_result = response.results[-1]
            confidence = last_result.alternatives[0].confidence if last_result.is_final else None

            if last_result.is_final:
                translation = translate_text(full_text)
                logger.log_result(full_text, confidence, True, translation=translation)
                print(f"\n✅ {full_text}")
                print(f"🌍 → {translation}\n\n\n\n")

                # interim キャッシュをリセット
                last_interim_text = ""
                last_interim_translation = ""

            else:
                # 同じテキストならスキップ
                if full_text == last_interim_text:
                    continue
                last_interim_text = full_text

                # 翻訳を更新
                translation = translate_text(full_text)
                last_interim_translation = translation

                logger.log_result(full_text, confidence, False, translation=translation)
                print_interim_block(full_text, translation)

    except Exception as e:
        print(f"⚠️ 認識エラー: {e}")



    recorder_thread.join()
    wav_path = f"{AUDIO_DIR}/{logger.session_id}.wav"
    logger.save(wav_path)
    print(f"💾 音声とログを保存: {wav_path}")

# === 実行 ===
if __name__ == "__main__":
    streaming_asr()

