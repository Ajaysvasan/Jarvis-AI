import faster_whisper
import os
import pyaudio 
import matplotlib.pyplot as plt
import numpy as np
import wave
import time 

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNEL = 1
RATE = 160000
# model = faster_whisper.WhisperModel("medium",compute_type="float16")

class SpeechToText:
    def __init__(self,model = None):
        os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
        self.chunck = 1024
        self.model = model
        self.p = pyaudio.PyAudio() 

    def processingAudioInFiles(self,filePath):
        obj = wave.open(filePath,"rb")
        sampleFreq = obj.getframerate()
        nSamples = obj.getnframes()
        signalWave = obj.readframes(-1)
        obj.close()
        return sampleFreq,nSamples,signalWave
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

    def microPhoneInput(self,framesPerBuffer,channel,format,rate):
        p =  pyaudio.PyAudio()

        stream = p.open(
            format=format,
            channels=channel,
            rate = rate,
            input = True,
            frames_per_buffer=framesPerBuffer
        )
        frames = []
        try:
            while True:
                data = stream.read(framesPerBuffer)
                frames.append(data)
        except KeyboardInterrupt:
            pass 
        stream.stop_stream()
        stream.close()
        p.terminate()

sst = SpeechToText()

audioFilePath = r"D:\SIH project\AUDIO\REAL\margot-original.wav"

sampleFreq,nSamples,signalWave = sst.processingAudioInFiles(audioFilePath)

sst.audioPlot(signalWave=signalWave, sampleFreq=sampleFreq, nSamples=nSamples)