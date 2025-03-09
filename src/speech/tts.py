import re
import unidecode
import inflect
import nltk

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
        self.text = preproceedText.strip()

class PhonemeMapper:
    def __init__(self):
        self.phoneme_map = self.build_phoneme_map()

    def build_phoneme_map(self):
        pass

    def apply_contextual_rules(self, word):
        pass

    def map_word_to_phonemes(self, word):
        pass

    def map_text(self, text):
        pass

t = TextPreprocessing("Hello world 1234 $ % . I'am Ajay s vasan and I'am a DR.")
# t.normalize()
print(t.abrevations())