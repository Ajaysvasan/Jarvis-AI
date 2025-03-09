import re
import unidecode
import inflect
import nltk
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

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

t = TextPreprocessing("Hello world.")

phoneme = PhonemeMapper()

print(phoneme.map_text(t.numbersToText()))