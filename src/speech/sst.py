import faster_whisper
import os
import pyaudio 
import matplotlib.pyplot as plt
import numpy as np
import wave
import silero_vad
import torch 
import io

FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNEL = 1
RATE = 16000
SILENCE_VALUE = 5

class SpeechToText:
    def __init__(self,model = None):
        os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
        self.chunck = 1024
        self.model = model if model else faster_whisper.WhisperModel("medium",compute_type="int8")

        self.vadModel , utils = torch.hub.load(
                                repo_or_dir='snakers4/silero-vad',
                                model = "silero_vad",
                                trust_repo =True,
                                force_reload=True)
        self.vadModel.eval()
        self.getSpeechTimeStamps = utils[0]

    def processingAudioInFiles(self,filePath):
        obj = wave.open(filePath,"rb")
        sampleFreq = obj.getframerate()
        nSamples = obj.getnframes()
        signalWave = obj.readframes(-1)
        obj.close()
        return sampleFreq,nSamples,signalWave
    
    def writeWaveFile(self):
        frames = self.recording()
        p = pyaudio.PyAudio()
        obj = wave.open("temp.wav","wb")
        obj.setnchannels(CHANNEL)
        obj.setsampwidth(p.get_sample_size(FORMAT))
        obj.setframerate(RATE)
        obj.writeframes(b''.join(frames))
        obj.close()
        
        
    def recording(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            rate = RATE,
            channels=CHANNEL,
            input = True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )
        print("Start recording")
        frames = []
        try:
            while True:
                data = stream.read(FRAMES_PER_BUFFER,exception_on_overflow=False)
                frames.append(data)
        except KeyboardInterrupt:
            print("Recording stopped")
            stream.stop_stream()
            stream.close()
            p.terminate()
        return frames
    

    def transcribing(self,audioFile):
        self.writeWaveFile()
        segments,_ = self.model.transcribe(audioFile)
        text = "".join([segment.text for segment in segments])
        return text

    # signalWaves -> total number of frames, signalFreq -> frameRate, nSamples -> number of samples (obj.getnframe)
    def audioPlot(self,signalWave,sampleFreq,nSamples):
        tAudio = nSamples / sampleFreq

        signalArray = np.frombuffer(signalWave,dtype=np.int16)

        times = np.linspace(0,tAudio,num= len(signalArray))

        plt.figure(figsize=(15,5))
        plt.plot(times,signalArray)

        plt.title("Audio Signal")
        plt.ylabel("Signal Wave")
        plt.xlabel("Time in sec ")
        plt.xlim(0,tAudio)
        plt.show()




audioFilePath = os.path.abspath("temp.wav")

sst = SpeechToText()
test = sst.transcribing(audioFile=audioFilePath)
print(test)