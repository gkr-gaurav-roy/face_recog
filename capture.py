# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:07:33 2024

@author: GAURAV KUMAR RAY
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


BASE_DIR = os.getcwd()

print(BASE_DIR)

cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)


img_count = 0
images = []

while img_count <= 300:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    faces = cascade.detectMultiScale(frame)
    if len(faces)>0:
        for x,y,w,h in faces:
            face_img = frame[y:y+h,x:x+w].copy()
            cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0))
            if cv2.waitKey(1) == ord('x'):
                img_count+=1
                cv2.putText(frame, f"saving face {img_count}", (10,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
                images.append(face_img)
                
                
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

name = input("enter name: ")

folder = os.path.join(BASE_DIR,name)
if not os.path.exists(folder):
    os.mkdir(folder)

for num,image in tqdm(enumerate(images)):
    cv2.imwrite(os.path.join(folder,f"{name}_{num}.jpg"), image)





