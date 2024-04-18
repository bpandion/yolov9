
import os

path='KichenCOCOv2-3\\valid\\labels\\old_labels'
txts = os.listdir(path)
if not os.path.exists(path+'\\edited'):
    os.makedirs(path+'\\edited')
filecounter=1
for txt in txts:
    with open(path+'\\'+txt, "r") as file:
        # annotation txt auslesen und zerlegen
        content =  file.readlines()
        split_content = []
        for line in content:
            split_content.append(line.split(' '))
        # class id ändern
        for line in split_content:
                if line[0] == '1': 
                    # apple 
                    # line[0] = '53'
                    line[0] = '47'
                elif line[0] == '2':
                    #banana
                    # line[0] = '52'
                    line[0] = '46'
                elif line[0] == '3':
                    #bottle
                    # line[0] = '44'
                    line[0] = '39'
                elif line[0] == '4':
                    #bowl
                    # line[0] = '51'
                    line[0] = '45'
                elif line[0] == '5':
                    #clock
                    # line[0] = '85'
                    line[0] = '74'
                elif line[0] == '6':
                    #cup
                    # line[0] = '47'
                    line[0] = '41'
                elif line[0] == '7':
                    #knife
                    # line[0] = '49'
                    line[0] = '43'
                elif line[0] == '8':    
                    #microwave
                    # line[0] = '78'
                    line[0] = '68'
                elif line[0] == '9':
                    #refrigerator
                    # line[0] = '82'
                    line[0] = '72'
                elif line[0] == '10':
                    #sink
                    # line[0] = '81'
                    line[0] = '71'
                elif line[0] == '11':
                    #spoon
                    # line[0] = '50'
                    line[0] = '44'
                elif line[0] == '12':
                    #toaster
                    # line[0] = '80'  
                    line[0] = '70' 
        
        # txt wieder zusammensetzen
        full_string = ''
        for s in split_content:
            full_string = full_string + ' '.join(s)
        
        # write editierte txt 
        # with open(path+'\\edited\\'+txt, "w") as file:
        with open('KichenCOCOv2-3\\valid\\labels\\'+txt, "w") as file:
            file.write(full_string)
            file.close()
        print(str(filecounter) + " File edited, Filename: " + '\\edited\\'+txt)
        filecounter = filecounter+1