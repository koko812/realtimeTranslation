# realtimeTranslation

### 準備
```
git clone https://github.com/koko812/realtimeTranslation.git
uv sync
```

### 声を出させる
```
# 英語
uv run streaming_asr_realtime_tts.py
# 中国語
uv run streaming_asr_realtime_tts.py --lang zh
```


### 文字起こしだけ
```
uv run streaming_asr_realtime_translation.py
```

