import argparse
from google.cloud import speech

# 音声ファイルを指定して，google の asr の API に投げて文字起こし結果を受け取るだけのコード
# wav 形式しか使えないことに注意

def transcribe_audio(audio_path: str):
    client = speech.SpeechClient()

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    audio = speech.RecognitionAudio(content=audio_data)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ja-JP",  # 日本語
    )

    response = client.recognize(config=config, audio=audio)

    print("send_audio")
    for result in response.results:
        print("認識結果:", result.alternatives[0].transcript)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Cloud 音声認識")
    parser.add_argument("audio_path", type=str, help="音声ファイルのパス（WAV形式）")
    args = parser.parse_args()

    transcribe_audio(args.audio_path)

