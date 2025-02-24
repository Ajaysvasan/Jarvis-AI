import whisper
import os

model = whisper.load_model("medium")

class SpeechToText:
    def __init__(self,model):
        os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
        self.model = model

    def recognize_speech(self,audioFile):
        return self.model.transcribe(audioFile)['text']


sst = SpeechToText(model)

audioFilePath = r"D:\SIH project\AUDIO\REAL\margot-original.wav"

result = sst.recognize_speech(audioFilePath)

print(f'Speech: {result}')