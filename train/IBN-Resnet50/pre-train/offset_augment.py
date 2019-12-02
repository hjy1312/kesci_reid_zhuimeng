from PIL import Image
import os
path='./train_set/train'
dirs=os.listdir(path)
for file in dirs:
    full_path=os.path.join(path,file)
    if len(os.listdir(full_path))<5:
        for lists in os.listdir(full_path):
            img_path=os.path.join(full_path,lists)
            img=Image.open(img_path)
            width=img.size[0]
            height=img.size[1]
            img2=img.crop((18,16,width,height))
            img2_path=os.path.join(full_path,lists[:-4]+'_1.png')
            img2.save(img2_path)
            img3 = img.crop((0, 0, width-18, height-16))
            img3_path = os.path.join(full_path, lists[:-4] + '_2.png')
            img3.save(img3_path)
            img4 = img.crop((0, 16, width - 18, height ))
            img4_path = os.path.join(full_path, lists[:-4] + '_3.png')
            img4.save(img4_path)
            img5 = img.crop((18, 0, width , height-16))
            img5_path = os.path.join(full_path, lists[:-4] + '_4.png')
            img5.save(img5_path)
