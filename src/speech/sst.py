import faster_whisper
import os
import pyaudio 

model = faster_whisper.WhisperModel("tiny",compute_type="float16")

class SpeechToText:
    def __init__(self,model):
        os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
        self.chunck = 1024
        self.model = model
        self.p = pyaudio.PyAudio()

    def recordChunck(p,stream,filePath,chuckLength = 1):
        frames = []
        for _ in range(0,int(16000/1024 * chuckLength)):
            data = stream.read(1024)
            frames.append(data)
        return ""
    def recognize_speech(self,audioFile):
        stream = self.p.open(format=pyaudio.paInt16,channels=1,rate = 16000,input = True,frames_per_buffer=1024)
        accumulatedTranscription = ""

                
        return self.model.transcribe(audioFile)['text']


sst = SpeechToText(model)

audioFilePath = r"D:\SIH project\AUDIO\REAL\margot-original.wav"

result = sst.recognize_speech(audioFilePath)

print(f'Speech: {result}')