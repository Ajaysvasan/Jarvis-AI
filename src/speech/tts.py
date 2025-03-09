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
        return preproceedText.strip()
    

t = TextPreprocessing("Hello world 1234 $%. I'am Ajay s vasan and I'am a DR.")

print(t.numbersToText())