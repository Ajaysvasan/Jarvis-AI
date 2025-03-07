from phonemizer import phonemize

text = "Hi, I am Jarvis"

phonemz = phonemize(text,language='en',backend='segments')

print(phonemz)