#!/usr/bin/env python
# coding: utf-8




import cv2
import numpy as np
import os 
import glob
from datetime import date, datetime
from recogFunc import read






recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

id = 0 #initialize userID 
UserNames = read('names.csv') #userList readed by read()
currTemp = [] #current person's temperature
currName = '' #current person's name
        
temp = 0; #initialize arbitrary temp

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)))
    
    #when detect the face
   
    #show name on the face rect
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #face area
        id, loss = recognizer.predict(gray[y:y+h,x:x+w]) #compute face id and loss
        # Check if loss of prediction, 0 is perfect match 
        if (loss < 90):
            currName = UserNames[id]
        else:
            currName = "unknown"
        #show current name
        if currName != 'unknown':
            cv2.putText(img, str(currName), (x+5,y-5), font, 1, (255,255,255), 2)
            
            #show temp, red if over 38, blue when normal
            
        else:
            cv2.putText(img, 'unauthorized', (x+5,y-5), font, 1, (0,0,255), 2)
        
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break            
    #write temp and date to the this person's temperature file
    if currName != 'unknown':
        currTemp.append(temp)
    if len(faces) == 0: #when the face is out of camera
        
            currName = 'unknown'
            
print("\n Program Exited")
cam.release()
cv2.destroyAllWindows()







