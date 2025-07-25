from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()
voices = client.list_voices()

for voice in voices.voices:
    langs = ", ".join(voice.language_codes)
    print(f"▶ {voice.name} | Lang: {langs} | Gender: {texttospeech.SsmlVoiceGender(voice.ssml_gender).name}")

