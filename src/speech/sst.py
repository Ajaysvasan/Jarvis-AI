import faster_whisper
import os
import pyaudio 
import matplotlib.pyplot as plt
import numpy as np
import wave
import silero_vad
import torch 

FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNEL = 1
RATE = 16000
model = faster_whisper.WhisperModel("medium",compute_type="int8")
SILENCE_VALUE = 5

class SpeechToText:
    def __init__(self,model = None):
        os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
        self.chunck = 1024
        self.model = model
        self.p = pyaudio.PyAudio() 

        self.vadModel , utils = torch.hub.load(
                                repo_or_dir='snakes4/silero_vad',
                                model = "silero_vad",
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

    def isSpeech(self,audioChuck):
        audio_int16 = np.frombuffer(audioChuck,np.int16)
        audio_float32 = torch.FloatTensor(audio_int16.astype(np.int16)/32768.0)
        getTimeStamp = self.getSpeechTimeStamps(
            audio_float32,
            self.vadModel,
            threshold = 0.5,
            sampling_rate = RATE
        )
        return len(getTimeStamp)>0

sst = SpeechToText()

audioFilePath = r"D:\SIH project\AUDIO\REAL\margot-original.wav"

# sampleFreq,nSamples,signalWave = sst.processingAudioInFiles(audioFilePath)

# sst.audioPlot(signalWave=signalWave, sampleFreq=sampleFreq, nSamples=nSamples)

sst.microPhoneInput(framesPerBuffer=FRAMES_PER_BUFFER,channel=CHANNEL,format=FORMAT,rate = RATE)