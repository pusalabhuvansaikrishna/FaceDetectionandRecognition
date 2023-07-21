import os
import cv2
from PIL import Image
import numpy as np
import pickle
model=cv2.face.LBPHFaceRecognizer_create()
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
base_dir=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(base_dir,"Images")
current_id=0
label_ids={}
x_train=[]
y_label=[]
for root,dir,files in os.walk(img_dir):
    for file in files:
        if (file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png")):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            if label in label_ids:
                pass
            else:
                label_ids[label]=current_id
                current_id+=1

            id_=label_ids[label]
            pil_image=Image.open(path).convert("L")
            final_image=pil_image.resize((550,550),Image.ANTIALIAS)
            image_array=np.array(final_image,"uint8")
            faces=face_cascade.detectMultiScale(image_array,1.5,5)
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_label.append(id_)

with open("label.pickle","wb") as file:
    pickle.dump(label_ids,file)

model.train(x_train,np.array(y_label))
model.save("trainner.yml")