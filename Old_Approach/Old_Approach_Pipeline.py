from sklearn.decomposition import FastICA
import cv2

from Second_module import *
from Fourth_Module import *
from Data_Read import *

import cv2

window_size = 150
window_slide = 25

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r'D:\Uni\GP\Dataset\id1\alex\alex_resting\cv_camera_sensor_stream_handler.avi')



(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
     
hr_count = 0
count = 0
sum_right = 0
images = []
first_time = 1
compare = window_size
window_step = window_slide
while True:
    # Read the frame
    _, img = cap.read()
    if img.any():

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w] #slice the face from the image
            count+=1
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
            images.append(face)
            if(count == compare):
                Matrix = ROItoRGBchannels(images)
                    
                transformer = FastICA(whiten=True)
                Sources = transformer.fit_transform(Matrix.T)
            
                hr_out = get_heart_rate(Sources,WINDOW_SIZE=window_size,FPS=window_slide)
            
                hr_count+=1
                count = 0
                images = images[window_step:]
                
                compare = window_step
        # Display
        cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k==27 or k==-1:
        break
cv2.destroyAllWindows()
cap.release()

