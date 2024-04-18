import os
import json
from roboflow import Roboflow

rf = Roboflow(api_key="0kmGWteb4DO0DqJvJg8T")
# download dataset in coco and yolov9 format

project = rf.workspace("lvarobotvision").project("kitchencocov2")
version = project.version(3)
dataset = version.download("coco")
os.rename('.\\KitchenCOCOv2-3','.\\KitchenCOCOv2-3_coco')

project = rf.workspace("lvarobotvision").project("kitchencocov2")
version = project.version(3)
dataset = version.download("yolov9")





master_path ='.\\KitchenCOCOv2-3'
coco_master_path = '.\\KitchenCOCOv2-3_coco'
old_label_path = master_path+'\\valid\\labels'
old_img_path = master_path+'\\valid\\images'

# build file structure 
if not os.path.exists(master_path+'\\annotations'):
    os.makedirs(master_path+'\\annotations')

if not os.path.exists(master_path+'\\images'):
    os.makedirs(master_path+'\\images')
if not os.path.exists(master_path+'\\images\\val2017'):
    os.makedirs(master_path+'\\images\\val2017')
    
if not os.path.exists(master_path+'\\labels'):
    os.makedirs(master_path+'\\labels')
if not os.path.exists(master_path+'\\labels\\val2017'):
    os.makedirs(master_path+'\\labels\\val2017')
    
new_label_path = master_path + '\\labels\\val2017'
new_img_path = master_path + '\\images\\val2017'
new_annotation_path = master_path + '\\annotations'

# save imgs in new dir
imgs = os.listdir(old_img_path)
for img in imgs:
    os.rename(old_img_path+'\\'+img,new_img_path+'\\'+img)

print('\n##### Fixing annotation json #####')
# move & rename annotations-json
os.rename(coco_master_path+'\\valid\\_annotations.coco.json',new_annotation_path+'\\instances_val2017.json')

json_file = new_annotation_path + '\\instances_val2017.json'
stng=''
count =0
# first id: id after coco dataset class list
# second id: id after coco.yaml class list

with open(json_file,'r') as anno_data:
        js = json.loads(anno_data.read())
        categories = js['categories']
        annotations = js['annotations']
        for cat in js['categories']:
            # id 0 = "Kitchen-Objects"
            if cat['id'] == 1: 
                # apple 
                # cat['id'] = 53
                cat['id'] = 47
                # cat['id'] = 1
            elif cat['id'] == 2:
                #banana
                # cat['id'] = 52
                cat['id'] = 46
                # cat['id'] = 2
            elif cat['id'] == 3:
                #bottle
                # cat['id'] = 44
                cat['id'] = 39
                # cat['id'] = 3
            elif cat['id'] == 4:
                #bowl
                # cat['id'] = 51
                cat['id'] = 45
                # cat['id'] = 4
            elif cat['id'] == 5:
                #clock
                # cat['id'] = 85
                cat['id'] = 74
                # cat['id'] = 5
            elif cat['id'] == 6:
                #cup
                # cat['id'] = 47
                cat['id'] = 41
                # cat['id'] = 6
            elif cat['id'] == 7:
                #knife
                # cat['id'] = 49
                cat['id'] = 43
                # cat['id'] = 7
            elif cat['id'] == 8:    
                #microwave
                # cat['id'] = 78
                cat['id'] = 68
                # cat['id'] = 8
            elif cat['id'] == 9:
                #refrigerator
                # cat['id'] = 82
                cat['id'] = 72
                # cat['id'] = 9
            elif cat['id'] == 10:
                #sink
                # cat['id'] = 81
                cat['id'] = 71
                # cat['id'] = 10
            elif cat['id'] == 11:
                #spoon
                # cat['id'] = 50
                cat['id'] = 44
                # cat['id'] = 11
            elif cat['id'] == 12:
                #toaster
                # cat['id'] = 80
                cat['id'] = 70
                # cat['id'] = 12
        
        for anno in js['annotations']:
            if anno['category_id'] == 1: 
                # apple 
                # anno['category_id'] = 53
                anno['category_id'] = 47
                # anno['category_id'] = 1
            elif anno['category_id'] == 2:
                #banana
                # anno['category_id'] = 52
                anno['category_id'] = 46
                # anno['category_id'] = 2
            elif anno['category_id'] == 3:
                #bottle
                # anno['category_id'] = 44
                anno['category_id'] = 39
                # anno['category_id'] = 3
            elif anno['category_id'] == 4:
                #bowl
                # anno['category_id'] = 51
                anno['category_id'] = 45
                # anno['category_id'] = 4
            elif anno['category_id'] == 5:
                #clock
                # anno['category_id'] = 85
                anno['category_id'] = 74
                # anno['category_id'] = 5
            elif anno['category_id'] == 6:
                #cup
                # anno['category_id'] = 47
                anno['category_id'] = 41
                # anno['category_id'] = 6
            elif anno['category_id'] == 7:
                #knife
                # anno['category_id'] = 49
                anno['category_id'] = 43
                # anno['category_id'] = 7
            elif anno['category_id'] == 8:    
                #microwave
                # anno['category_id'] = 78
                anno['category_id'] = 68
                # anno['category_id'] = 8
            elif anno['category_id'] == 9:
                #refrigerator
                # anno['category_id'] = 82
                anno['category_id'] = 72
                # anno['category_id'] = 9
            elif anno['category_id'] == 10:
                #sink
                # anno['category_id'] = 81
                anno['category_id'] = 71
                # anno['category_id'] = 10
            elif anno['category_id'] == 11:
                #spoon
                # anno['category_id'] = 50
                anno['category_id'] = 44
                # anno['category_id'] = 11
            elif anno['category_id'] == 12:
                #toaster
                # anno['category_id'] = 80
                anno['category_id'] = 70
                # anno['category_id'] = 12
            
        for i in categories:
            stng = stng + i['name'] + ', '
            count=count+1
        
        print('\n')
        print('\n')
        print('Annotaion-classes:\n')
        print(stng)
        s= '\n' + 'Nr of classes:' + str(count)
        print(s)
        print('\n')
        print(json.dumps(js['categories']))        

with open(json_file,'w') as file:
    json.dump(js,file)
    
print('\n##### Fixing Labels #####')
# fix lables
txts = os.listdir(old_label_path)
filecounter=1
for txt in txts:
    with open(old_label_path+'\\'+txt, "r") as file:
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
                    # line[0] = '1'
                elif line[0] == '2':
                    #banana
                    # line[0] = '52'
                    line[0] = '46'
                    # line[0] = '2'
                elif line[0] == '3':
                    #bottle
                    # line[0] = '44'
                    line[0] = '39'
                    # line[0] = '3'
                elif line[0] == '4':
                    #bowl
                    # line[0] = '51'
                    line[0] = '45'
                    # line[0] = '4'
                elif line[0] == '5':
                    #clock
                    # line[0] = '85'
                    line[0] = '74'
                    # line[0] = '5'
                elif line[0] == '6':
                    #cup
                    # line[0] = '47'
                    line[0] = '41'
                    # line[0] = '6'
                elif line[0] == '7':
                    #knife
                    # line[0] = '49'
                    line[0] = '43'
                    # line[0] = '7'
                elif line[0] == '8':    
                    #microwave
                    # line[0] = '78'
                    line[0] = '68'
                    # line[0] = '8'
                elif line[0] == '9':
                    #refrigerator
                    # line[0] = '82'
                    line[0] = '72'
                    # line[0] = '9'
                elif line[0] == '10':
                    #sink
                    # line[0] = '81'
                    line[0] = '71'
                    # line[0] = '10'
                elif line[0] == '11':
                    #spoon
                    # line[0] = '50'
                    line[0] = '44'
                    # line[0] = '11'
                elif line[0] == '12':
                    #toaster
                    # line[0] = '80'  
                    line[0] = '70' 
                    # line[0] = '12'
        
        # txt wieder zusammensetzen
        full_string = ''
        for s in split_content:
            full_string = full_string + ' '.join(s)
        
        # write editierte txt 
        # with open(path+'\\edited\\'+txt, "w") as file:
        with open(new_label_path+'\\'+txt, "w") as file:
            file.write(full_string)
            file.close()
        print(str(filecounter) + " File edited, Filename: " + new_label_path+'\\'+txt)
        filecounter = filecounter+1



val_img_string =''
imgs = os.listdir(new_img_path)
for img in imgs:
    val_img_string=val_img_string +'./images/val2017/'+img+'\n'

with open(master_path+'\\val2017.txt','w') as file:
    file.write(val_img_string)
    file.close()
    
print("DONE!")
print("You can now delete .\\KitchenCocov2-3_coco, it's no longer used.")