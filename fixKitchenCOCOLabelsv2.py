import os
import json,yaml
import numpy as np
from roboflow import Roboflow

rf = Roboflow(api_key="0kmGWteb4DO0DqJvJg8T")
# download dataset in coco and yolov9 format

project = rf.workspace("lvarobotvision").project("kitchencocov2")
version = project.version(5)
dataset = version.download("yolov9")


master_path ='.\\KitchenCOCOv2-5'
old_label_path = master_path+'\\valid\\labels'

    
if not os.path.exists(master_path+'\\new_labels'):
    os.makedirs(master_path+'\\new_labels')
    
new_label_path = master_path + '\\new_labels'


print('\n##### Fixing Labels #####')
# fix lables
txts = os.listdir(old_label_path)
filecounter=1
with open('data/coco.yaml','r') as file:
    coco_yaml_data = yaml.safe_load(file)

    
with open('KitchenCOCOv2-5/data.yaml','r') as file:
    kitchencoco_yaml_data = yaml.safe_load(file)
coco_names = coco_yaml_data['names']
kitchencoco_names = kitchencoco_yaml_data['names']

for txt in txts:
    with open(old_label_path+'\\'+txt, "r") as file:
        # annotation txt auslesen und zerlegen
        content =  file.readlines()
        split_content = []
        for line in content:
            split_content.append(line.split(' '))
        # class id ändern
        for line in split_content:
            classname = kitchencoco_names[int(line[0])]
            for ci in coco_names:
                if (coco_names[ci] == classname):
                    line[0] = str(ci)
                    break
            
    # txt wieder zusammensetzen
    
    full_string = ''
    for s in split_content:
        full_string = full_string + ' '.join(s)
    
    # write editierte txt 
    # with open(path+'\\edited\\'+txt, "w") as file:
    with open(new_label_path+'\\'+txt, "w") as file:
        file.write(full_string)
    print(str(filecounter) + " File edited, Filename: " + new_label_path+'\\'+txt)
    filecounter = filecounter+1
        
# editing yaml        
with open('KitchenCOCOv2-5/data.yaml','w') as file:
    new_yaml_data = kitchencoco_yaml_data
    new_yaml_data['names'] = coco_names
    yaml.dump(new_yaml_data,file)
    
    
print("DONE!")
print("NEXT MANUAL STEPS:")
print("-Delete old Labels in .\\KitchenCocov2-5\\valid\\labels.")
print("-Copy and paste new Labels from .\\KitchenCOCOv2-5\\new_labels to .\\KitchenCocov2-5\\valid\\labels.")
