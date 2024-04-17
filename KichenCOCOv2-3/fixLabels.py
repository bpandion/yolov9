
import os


txts = os.listdir('KichenCOCOv2-3\\valid\\labels')

for txt in txts:
    with open(txt, "w") as file:
        content =  file
        print(txt+"\n")