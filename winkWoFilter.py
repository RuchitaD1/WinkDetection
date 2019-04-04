#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os


from os import listdir
from os.path import isfile, join
import sys


# In[2]:


def detectWink(frame, location, ROI, cascade,c,d,right=False):
    
    #ROI = cv2.equalizeHist(ROI)
    #cv2.imshow("right",ROI)
    #cv2.waitKey(0)
    eyes = cascade.detectMultiScale(ROI,c,d,  0|cv2.CASCADE_SCALE_IMAGE, (5,10 ))
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        if right:
            x_r,y_r=ROI.shape
            # print(location,ROI.shape)
            x_r = int(x_r*7/8)
            cv2.rectangle(frame, (x+x_r,y), (x+w+x_r,y+h), (0, 255, 255), 2)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 2)
    return len(eyes)>0    # number of eyes is one


# In[3]:


def detect(frame, faceCascade, eyesCascade,c,d):
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # possible frame pre-processing:
    # gray_frame = cv2.equalizeHist(gray_frame)
    #gray_frame = cv2.medianBlur(gray_frame1, 5)

    scaleFactor = 1.05 # range is from 1 to ..
    minNeighbors = 10 # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (30,30) # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)
#     faces = face_cascade.detectMultiScale(gray_frame, 1.2, 5)
    detected = 0
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCount=0
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROInose = gray_frame[y:y+h, x:x+w]
        nose=cv2.CascadeClassifier('Downloads/Nariz.xml')
        noseno=nose.detectMultiScale(faceROInose,1.01,3,flag,minSize)
        #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        #gray_frame = cv2.filter2D(gray_frame, -1, kernel)
        if len(noseno)!=0:
            faceROI_L = gray_frame[y:int(y+(h*4/7)), x+int(w*0.1):x+int(w*0.52)]
            faceROI_R = gray_frame[y:int(y+(h*4/7)), x+int(w*0.5):(x+w)]
            '''cv2.imshow("left",faceROI_L)
            cv2.waitKey(0)
            cv2.imshow("right",faceROI_R)
            cv2.waitKey(0)'''
            eyeCount = 0

            eyeL = detectWink(frame, (x, y), faceROI_L, eyesCascade,c,d)

            if eyeL :
                 cv2.rectangle(frame, (x,y), (x+int(w*0.52),int(y+(h*4/7))), (0, 0, 255), 2)

            eyeR = detectWink(frame, (x, y), faceROI_R, eyesCascade,c,d,right=True)
            if eyeR :
                cv2.rectangle(frame, (x+int(w*0.5),y), ((x+w),int(y+(h*4/7))), (0, 0, 255), 2)
            eyeCount = eyeL + eyeR
            faceCount +=1
            # print("for face ",faceCount," Number of eyes = ",eyeCount)
            if eyeCount == 1:
                detected += 1
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                cv2.putText(frame,'Wink detected',(x,y), font, 0.5,(255,255,255),2,cv2.LINE_AA)

            else:

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    return detected


# In[4]:


def run_on_folder(cascade1, cascade2, folder,c,d):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]
    dt={}
    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f)
        r=700.0/img.shape[1]
        dim=(700,int(img.shape[0]*r))
        
        img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
        if type(img) is np.ndarray:
            
            lCnt = detect(img, cascade1, cascade2,c,d)
            dt[f]=lCnt
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount,dt


# In[5]:


def runonVideo(face_cascade, eyes_cascade,c,d):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade,c,d)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


# In[6]:


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    '''if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()'''

    # load pretrained cascades
    f=open('x.txt',mode='w')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_eye.xml')#_tree_eyeglasses
    option=input("1 or 2?")
    
    if option=="1":
        
        folderName = "Downloads/testImages"
        detections,dt = run_on_folder(face_cascade, eye_cascade, folderName,1.05,20)
        print("detections:"+str(detections))
      
        print(dt)
                
        '''   
        for k,v in detections:
            x=str(k)+","+str(v)+":"+str(detections[(k,v)])
            print(x)

            f.write(x)
        f.close()'''
        
        
    else: # no arguments'''
        runonVideo(face_cascade, eye_cascade,1.34,15)



# In[12]:


q


# In[ ]:





# In[ ]:




