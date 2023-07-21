import cv2
import pickle
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model=cv2.face.LBPHFaceRecognizer_create()
model.read("trainner.yml")
labels={}
with open("label.pickle",'rb') as file:
    og_labels=pickle.load(file)
    labels={v:k for k,v in og_labels.items()}
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame,1.5,5)
    for(x,y,w,h) in faces:
        roi_gray=gray_frame[y:y+h,x:x+w]
        id_,conf=model.predict(roi_gray)
        if (conf>=45 and conf<=85):
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            cv2.putText(frame,name,(x,y),font,1,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow("Video_live",frame)
    if cv2.waitKey(20)==ord("q"):
        break
cv2.destroyAllWindows()
