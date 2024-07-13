# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:37:40 2024

@author: GAURAV KUMAR RAY
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class model:
    
    def predict(img):
        return "prediction"

# load your own model here
model = model()


cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    faces = cascade.detectMultiScale(frame)
    if len(faces)>0:
        for x,y,w,h in faces:
            face_img = frame[y:y+h,x:x+w].copy()
            cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0))
            prediction = model.predict()
            cv2.putText(frame, f"predicted face {prediction}", (10,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
                
                
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()