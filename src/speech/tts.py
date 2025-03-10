import re
import unidecode
import inflect
import nltk
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# text -> normalize -> phonemes -> spectormes -> speech

import torchaudio

class TextPreprocessing:
    def __init__(self,text:str):
        self.text = text
        
    
    def normalize(self):
        self.text = self.text.lower()
        self.text = unidecode.unidecode(self.text)
        self.text = re.sub(r"[^\w\s]", "", self.text)

    def numbersToText(self):
        self.normalize()
        preproceedText = ""
        p = inflect.engine()
        for string in self.text.split():
            if string.isdigit():
                preproceedText += p.number_to_words((int(string))) + ' '
            else:
                preproceedText += string + " "
        return preproceedText.strip()

class PhonemeMapper:
    def __init__(self):
        self.phoneme_map = self.build_phoneme_map()
        self.default_phoneme_duration = {
            "AH": 0.08, "B": 0.07, "CH": 0.09, "D": 0.06, "EH": 0.07, "F": 0.07,
            "G": 0.07, "HH": 0.06, "IH": 0.07, "JH": 0.08, "K": 0.07, "L": 0.07,
            "M": 0.07, "N": 0.07, "OW": 0.08, "P": 0.07, "R": 0.08, "S": 0.07,
            "SH": 0.09, "T": 0.06, "TH": 0.08, "DH": 0.08, "UH": 0.07, "V": 0.07,
            "W": 0.07, "Y": 0.07, "Z": 0.07, "KS": 0.08, "KW": 0.08, " ": 0.1  # pause
        }


    def build_phoneme_map(self):
        
        digraphMap = {
            "ch": "CH",
            "sh": "SH",
            "th": {"voiceless": "TH", "voiced": "DH"},
            "ph": "F",
            "ck": "K",
            "gh": "G",
            "wh": "W",
            "kn": "N",
            "wr": "R",
            "qu": "KW"
        }

        singlLetteRmap = {
            "a": "AH",
            "b": "B",
            "c": {"default": "K", "soft": "S"},
            "d": "D",
            "e": "EH",
            "f": "F",
            "g": {"default": "G", "soft": "J"},
            "h": "HH",
            "i": "IH",
            "j": "JH",
            "k": "K",
            "l": "L",
            "m": "M",
            "n": "N",
            "o": "OW",
            "p": "P",
            "q": "K",
            "r": "R",
            "s": "S",
            "t": "T",
            "u": "UH",
            "v": "V",
            "w": "W",
            "x": "KS",
            "y": "Y",
            "z": "Z"
        }

        return {
            "digraphs": digraphMap,
            "singles": singlLetteRmap
        }


    def apply_contextual_rules(self, word:str):
        pass

    def assign_phoneme_durations(self,phoneme_sequence):
        durations = []
        for phoneme in phoneme_sequence:
            duration = self.default_phoneme_duration.get(phoneme,0.07)
            durations.append((phoneme,duration))
        return durations

    def get_th_phoneme(self, word, index):
        voicedWords = ["this", "that", "the", "those", "these"]
        if word in voicedWords:
            return self.phoneme_map["digraphs"]["th"]["voiced"]
        return self.phoneme_map["digraphs"]["th"]["voiceless"]

    def map_word_to_phonemes(self, word: str) -> list:
        phonemes = []
        i = 0
        while i < len(word):
            if i + 1 < len(word):
                digraph = word[i:i+2]
                if digraph in self.phoneme_map["digraphs"]:
                    if digraph == "th":
                        phoneme = self.get_th_phoneme(word, i)
                    else:
                        phoneme = self.phoneme_map["digraphs"][digraph]
                    phonemes.append(phoneme)
                    i += 2
                    continue
            
            char = word[i]
            next_char = word[i+1] if i+1 < len(word) else ""

            if char in self.phoneme_map["singles"]:
                if char == "c":
                    if next_char in ['e', 'i', 'y']:
                        phoneme = self.phoneme_map["singles"]["c"]["soft"]
                    else:
                        phoneme = self.phoneme_map["singles"]["c"]["default"]
                elif char == "g":
                    if next_char in ['e', 'i', 'y']:
                        phoneme = self.phoneme_map["singles"]["g"]["soft"]
                    else:
                        phoneme = self.phoneme_map["singles"]["g"]["default"]
                else:
                    phoneme = self.phoneme_map["singles"][char]
            else:
                phoneme = char.upper() 
            
            phonemes.append(phoneme)
            i += 1
        
        return phonemes

    def map_text(self, text):
        words = text.split()
        return {
            word:self.map_word_to_phonemes(word) for word in words
        }


class AcousticFeatureExtractor:
    def __init__(self, sampleRate=16000, nMels=80, nFft=1024, hopLength=256):
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.nFft = nFft
        self.hopLength = hopLength

    def load_audio(self, filePath):
        return librosa.load(filePath,sr = self.sampleRate)

    def extract_mel_spectrogram(self, waveform):
        melSpec = librosa.feature.melspectrogram(
            y= waveform,
            sr = self.sampleRate,
            n_fft=self.nFft,
            hop_length=self.hopLength,
            n_mels = self.nMels
        )

        melSpecDb = librosa.power_to_db(melSpec,ref=np.max)

        return melSpecDb
    
    def inverted_mel_spectrogram(self,melSpectrogramDb):
        melSpectrogram = librosa.db_to_power(melSpectrogramDb)
        
        melFilter = librosa.filters.mel(
            sr=self.sampleRate,
            n_fft=self.nFft,
            n_mels=self.nMels
        )

        invMelFilter = np.linalg.pinv(melFilter)

        magnitude = np.dot(invMelFilter,melSpectrogram)

        waveform = librosa.griffinlim(
            magnitude,
            n_iter=32,
            hop_length=self.hopLength,
            win_length=self.nFft
        )

        return waveform
    
    def visualize_mel_spectrogram(self, mel_spec_db):
        librosa.display.specshow(mel_spec_db, sr=self.sampleRate, hop_length=self.hopLength, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.show()

    def write_audio(self,waveform):
        sf.write("reconstructed_output.wav", waveform, self.sampleRate)

t = TextPreprocessing("Anish Raj.")



phoneme = PhonemeMapper()

features = AcousticFeatureExtractor()

mapedText = phoneme.map_text(t.numbersToText())

print(phoneme.assign_phoneme_durations(mapedText))

FILE_PATH = r'd:\SIH project\AUDIO\REAL\linus-original.wav'

print(features.load_audio(FILE_PATH))

waveform,sampleRate = features.load_audio(FILE_PATH)

print(sampleRate)

print(features.extract_mel_spectrogram(waveform))
