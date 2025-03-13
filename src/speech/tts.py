# text -> normalize -> phonemes -> spectrogram -> speech
import librosa.display
import re
import unidecode
import inflect
import nltk
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import torch
import json
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn


EPOCHS = 50

FILE_PATH = r'temp.wav'
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

    def apply_contextual_rules(self, word: str) -> str:
        if word.startswith("kn"):
            word = word.replace("k", "", 1)
        if word.startswith("wr"):
            word = word.replace("w", "", 1)
        if word.startswith("gn"):
            word = word.replace("g", "", 1)
        if word.endswith("mb"):
            word = word[:-1]

        word = word.replace("ai", "ay")
        word = word.replace("ea", "ee")
        word = word.replace("ie", "iy")
        word = word.replace("oo", "uw")
        word = word.replace("ou", "ow")
        word = word.replace("igh", "ay") 

        custom_exceptions = {
            "knight": "nite",
            "write": "rite",
            "comb": "cohm",
            "sword": "sord",
            "colonel": "kernel",
            "often": "offen"
        }
        if word in custom_exceptions:
            return custom_exceptions[word]

        return word


    def assign_phoneme_durations(self,phoneme_sequence):
        durations = []
        for phoneme in phoneme_sequence:
            duration = self.default_phoneme_duration.get(phoneme,0.07)
            durations.append((phoneme,duration))
        return durations

    def get_th_phoneme(self, word,index):
        voicedWords = ["this", "that", "the", "those", "these"]
        if word in voicedWords:
            return self.phoneme_map["digraphs"]["th"]["voiced"]
        return self.phoneme_map["digraphs"]["th"]["voiceless"]

    def map_word_to_phonemes(self, word: str) -> list:
        word = self.apply_contextual_rules(word)
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

    def duration_to_frame_mapping(self,phonemeDurationList,hopLength = 256,sampleRate = 16000):
        totalDuration = sum([dur for _,dur in phonemeDurationList])
        totalSamples = totalDuration * sampleRate
        totalFrames = int(totalSamples/hopLength)

        totalPhonemeDuration = sum([dur for _,dur in phonemeDurationList])
        frameMapping = []
        for phoneme,dur in phonemeDurationList:
            frames = int((dur / totalPhonemeDuration) * totalFrames)
            frameMapping.append((phoneme,frames))

        return frameMapping

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
    
    def visualize_phoneme_alignment(self,melSpecDb,frameMapping,hopLength = 256, sampleRate = 16000 ):
        librosa.display.specshow(melSpecDb,sr = sampleRate, hop_length=hopLength,x_axis='time',y_axis='mel')
        plt.colorbar(format='%+2.0f db')
        frameIndex = 0
        for phoneme,frameCount in frameMapping:
            time = frameIndex * hopLength / sampleRate
            plt.axvline(x=time,color='r',linestyle = '--',linewidth = 0.7) 
            plt.text(time,melSpecDb.shape[0] + 5,phoneme,rotation = 90,verticalalignment = 'bottom',fontsize = 8)
            frameIndex+=frameCount

        plt.title("Phoneme Alignment Over Mel-Spectrogram")
        plt.tight_layout()
        plt.show()
    
    def visualize_mel_spectrogram(self, mel_spec_db):
        librosa.display.specshow(mel_spec_db, sr=self.sampleRate, hop_length=self.hopLength, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.show()

    def write_audio(self,waveform):
        sf.write("reconstructed_output.wav", waveform, self.sampleRate)


class Vocoder:
    def __init__(self,sampleRate = 16000):
        self.sampleRate = sampleRate

    def griffin_lim_vocoder(self,melSpecDb):
        melPower = librosa. db_to_power(melSpecDb)
        melFilter = librosa.filters.mel(sr = self.sampleRate, n_fft=1024,n_mels=80)
        invMelFilter = np.linalg.pinv(melFilter)
        magnitude = np.dot(invMelFilter,melPower)

        waveform = librosa.griffinlim(magnitude,n_iter=32,hop_length=256,win_length=1024)

        return waveform
        
    def save_waveform(self,waveform,filename = "output.wav"):
        sf.write(filename,waveform,self.sampleRate)
class TTS:
    def __init__(self):
        self.textPreprocessing = TextPreprocessing("")
        self.phonemeMapper = PhonemeMapper()
        self.featureExtractor = AcousticFeatureExtractor()
        self.vocoder = Vocoder()

    def synthesis(self,text:str,outputFileName = "output.wav",visualize = True):
        self.textPreprocessing.text = text
        normalizedText = self.textPreprocessing.numbersToText()

        phonemeSequence = []
        for word in normalizedText.split():
            phonemeSequence.extend(self.phonemeMapper.map_word_to_phonemes(word) + [' '])
        phonemeDurations = self.phonemeMapper.assign_phoneme_durations(phonemeSequence)
        frameMapping = self.phonemeMapper.duration_to_frame_mapping(phonemeDurations)

        dummyWaveform = np.random.rand(16000)

        melSpec =  self.featureExtractor.extract_mel_spectrogram(dummyWaveform)
        reconstructedWave = self.vocoder.griffin_lim_vocoder(melSpec)

        self.vocoder.save_waveform(reconstructedWave,filename = outputFileName)
        if visualize:
            self.featureExtractor.visualize_mel_spectrogram(melSpec)
        return outputFileName
    

class PhonemeVocabularyBuilder:
    def __init__(self,phonemeMapper:dict):
        self.phonemeMapper = phonemeMapper

    def build_vocab(self,save_path = 'phoneme_vocab.json'):
        uniquePhoneme = set()

        for val in self.phonemeMapper.phoneme_map['singles'].values():
            if isinstance(val,dict):
                uniquePhoneme.update(val.values())

            else:
                uniquePhoneme.add(val)

        for val in self.phonemeMapper.phoneme_map['digraphs'].values():
            if isinstance(val,dict):
                uniquePhoneme.update(val.values())

            else:
                uniquePhoneme.add(val)

        uniquePhoneme.add(" ")

        vocabList = ["<PAD>", "<UNK>"] + sorted(list(uniquePhoneme))

        phonemeToId = {phoneme: idx for idx,phoneme in enumerate(vocabList)}

        with open(save_path,'w') as file:
            json.dump(phonemeToId,file,indent=4)

        return phonemeToId

class PhonemeToTensorConverter:
    def __init__(self,vocabPath = "phoneme_vocab.json"):
        with open(vocabPath,'r') as file:
            self.phonemeToId = json.load(file)
        self.idToPhoneme = {v:k for k,v in self.phonemeToId.items()}
        self.padId = self.phonemeToId.get("<PAD>",0)
        self.unkToId = self.phonemeToId.get("<UNK>",1)

    def phoneme_to_tensor(self,phonemeSequence,maxLength = None):
        ids = [self.phonemeToId.get(p,self.unkToId) for p in phonemeSequence]
        if maxLength:
            ids = ids[:maxLength] + [self.padId] * max(0,maxLength - len(ids))
        return torch.tensor(ids, dtype=torch.long)
        
    def tensor_to_phonemes(self,idTensor):
        return [self.idToPhoneme.get(int(idx),"<UNK>") for idx in idTensor]
        
    def __len__(self):
        return len(self.phonemeToId)
    
class TTSDataset(Dataset):
    def __init__(self,phonemeSequence, melSpecs,converter,maxPhonemeLength = None,maxMelSpecLength = None):
        self.phonemeSequence = phonemeSequence
        self.melSpecs = melSpecs
        self.converter = converter
        self.maxPhonemeLength = maxPhonemeLength
        self.maxMelSpecLength = maxMelSpecLength
    
    def __len__(self):
        return len(self.phonemeSequence)

    def __getitem__(self, idx):
        phoneme = self.phonemeSequence[idx]
        melSpec = self.melSpecs[idx]
        phonemeTensor = self.converter.phoneme_to_tensor(phoneme,maxLength = self.maxPhonemeLength)
        melTensor = torch.tensor(melSpec,dtype = torch.float32)


        if self.maxPhonemeLength:
            timeDim = melTensor.shape[1]
            if timeDim < self.maxPhonemeLength:
                pad = self.maxPhonemeLength - timeDim
                melTensor = torch.nn.functional.pad(melTensor,(0,pad),mode = 'constant',value=0 )
            else:
                melTensor = melTensor[:,:self.maxPhonemeLength]
        

        return phonemeTensor,melTensor
    
class PhonemeToMelModel(nn.Module):
    def __init__(self, vocabSize, embeddigDim = 128,hiddenDim=256,numLayers = 8,melBins = 80):
        super(PhonemeToMelModel,self).__init__()
        self.embedding = nn.Embedding(vocabSize,embedding_dim=embeddigDim,padding_idx=0)
        self.lstm = nn.LSTM(input_size=embeddigDim,
                            hidden_size=hiddenDim,
                            num_layers=numLayers,
                            batch_first=True,
                            bidirectional=True)
        
        self.fc = nn.Linear(hiddenDim * 2, melBins)

    def forward(self,X):
        x = self.embedding(X)
        lstmOut,_ = self.lstm(x)
        melOut = self.fc(lstmOut)

        return melOut


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self,model,dataLoader,epochs = 50,lr=1e-3,modelPath = 'phonemeToModel.pth'):
        self.model = model
        self.model.to(DEVICE)
        self.dataLoader = dataLoader
        self.epochs = epochs
        self.lr = lr
        self.modelPath = modelPath
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.criterion = nn.MSELoss()

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            epochLoss = 0.0
            for phonemeBatch,melBatch in self.dataLoader:
                phonemeBatch,melBatch = phonemeBatch.to(DEVICE),melBatch.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(phonemeBatch)
                output = output.transpose(1,2)
                if melBatch.shape[2] > output.shape[2]:
                    melBatch = melBatch[:,:output.shape[2]]
                elif melBatch.shape[2] < output.shape[2]:
                    pad = output.shape[2] - melBatch.shape[2]
                    melBatch = torch.nn.functional.pad(melBatch,(0,pad),mode = 'constant',value=0)
                loss = self.criterion(output,melBatch)
                loss.backward()
                self.optimizer.step()
                epochLoss+=loss.item()
            print(f'Epoch {epoch} | Loss: {epochLoss/len(self.dataLoader):.4f}')
        self.save()

    def save(self):
        torch.save(self.model.state_dict(),self.modelPath)
        print("Model has been saved")

    def load(self):
        self.model.load_state_dict(torch.load(self.modelPath))
        self.model.eval()
        print("model has been loaded")

def infer(text,model,converter,phonemeMapper,featureExtractor,vocoder,maxPhonemeLength = 30):
    textPre = TextPreprocessing(text)
    normText = textPre.numbersToText()

    phonemeSeq = []
    for word in normText.split():
            phonemeSeq.extend(phonemeMapper.map_word_to_phonemes(word) + [' '])

    inputTensor = converter.phoneme_to_tensor(phonemeSeq,maxLength = maxPhonemeLength).unsqueeze(0)
    inputTensor = inputTensor.to(DEVICE)
    model.eval()

    with torch.no_grad():
        melPred = model(inputTensor)
        melPred = melPred.squeeze(0).transpose(0,1)

    melPredNp = melPred.detach().cpu().numpy()
    waveform = vocoder.griffin_lim_vocoder(melPredNp)
    vocoder.save_waveform(waveform, filename='inference_output.wav')
    print("Audio saved successfully")


if __name__ == "__main__":
    texts = [
        "Hello world.",
        "Ajay is building his own TTS system.",
        "Deep learning is powerful.",
        "How are you doing today?"
    ]

    features = AcousticFeatureExtractor()
    phonemeMapper = PhonemeMapper()
    vocabBuilder = PhonemeVocabularyBuilder(phonemeMapper)
    vocab = vocabBuilder.build_vocab("phoneme_vocab.json")
    converter = PhonemeToTensorConverter("phoneme_vocab.json")

    phoneme_sequences = []
    mel_specs = []

    for text in texts:
        textPre = TextPreprocessing(text)
        normText = textPre.numbersToText()
        phoneme_seq = []
        for word in normText.split():
            phoneme_seq.extend(phonemeMapper.map_word_to_phonemes(word) + [' '])
        phoneme_sequences.append(phoneme_seq)

        # Dummy audio for now (load the same temp.wav for all samples)
        waveform, _ = features.load_audio(FILE_PATH)
        mel_spec = features.extract_mel_spectrogram(waveform)
        mel_specs.append(mel_spec)

    # Dataset + Dataloader
    dataset = TTSDataset(phoneme_sequences, mel_specs, converter, maxPhonemeLength=30, maxMelSpecLength=80)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model and Trainer
    model = PhonemeToMelModel(vocabSize=len(converter))
    trainer = Trainer(model, loader, epochs=EPOCHS, modelPath="phonemeToMelModel.pth")
    trainer.train()

    # Inference Test
    trainer.load()
    infer("Welcome to Jarvis AI", model, converter, phonemeMapper, features, Vocoder())
