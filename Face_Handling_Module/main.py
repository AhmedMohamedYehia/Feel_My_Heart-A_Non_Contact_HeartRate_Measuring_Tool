import cv2
import numpy as np 
from FaceDetection import CascadeClassifier, ViolaAndJones,WeakClassifier,HaarFeature,Rectangle
from FaceTracking import getHarrisCorners , featuresTranslation , applyGeometricTransformation
import time

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    # To use a video file as input 
    # cap = cv2.VideoCapture('filename.mp4')
    clf = CascadeClassifier.load("cascade")   
    TRACK = True
    enterr = False
    startXs,startYs = 0,0
    while True:
        # Read the frame
        _, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        graay = cv2.resize(gray,(320,240))
        imagee = cv2.resize(img,(320,240))

        # Detect the faces
        if TRACK:
            frame_start = time.time()
            faces = clf.get_face_bounding_box(input_img=graay,window_steps=10,pixel_steps=13,lower_window_range = 90, upper_window_range = 121)
            print("frame time: "+str(time.time() - frame_start))
            print(faces)
            if faces[0] != 0:
                bboxs = np.array([[faces[0],faces[1]],[faces[2],faces[1]],[faces[0],faces[3]],[faces[2],faces[3]]]).astype(float)
                startXs,startYs = getHarrisCorners(graay,bboxs)
                cv2.rectangle(img, (faces[0]*2, faces[1]*2), (faces[2]*2, faces[3]*2), (255, 0, 0), 2)
                cv2.imshow('image', img)
                TRACK = False
                enterr= True
                frame = imagee
        if enterr:
            frame_startt = time.time()
            newXs, newYs = featuresTranslation(startXs, startYs, frame, imagee)
            Xs, Ys ,bboxsn = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs)
            print("frame time: "+str(time.time() - frame_startt))
            startXs = Xs
            startYs = Ys

            if Xs.shape[0] < 15:
                try:
                    startXs,startYs = getHarrisCorners(cv2.cvtColor(imagee,cv2.COLOR_RGB2GRAY),bboxsn)
                except:
                    TRACK = True
                    enterr = False
            
            
            cv2.rectangle(img, (int(bboxsn[0][0]*2), int(bboxsn[0][1]*2)), (int(bboxsn[3][0]*2), int(bboxsn[3][1]*2)), (255, 0, 0), 2)
            cv2.imshow('img', img)
            bboxs = bboxsn
            frame = imagee

        # Stop if escape key is pressed
        k = cv2.waitKey(1)
        if k%256 == 27:
            break

        if k%256 == 32:
            TRACK = True
            enterr = False
            
    # Release the VideoCapture object
    cap.release()
    # while True:
    #     # Read the frame
    #     _, img = cap.read()

    #     # Convert to grayscale
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     graay = gray
    #     imagee = img

    #     # Detect the faces
    #     if TRACK:
    #         frame_start = time.time()
    #         faces = clf.get_face_bounding_box(input_img=graay,window_steps=20,pixel_steps=30,lower_window_range = 180, upper_window_range = 241)
    #         print("frame time: "+str(time.time() - frame_start))
    #         print(faces)
    #         if faces[0] != 0:
    #             bboxs = np.array([[faces[0],faces[1]],[faces[2],faces[1]],[faces[0],faces[3]],[faces[2],faces[3]]]).astype(float)
    #             startXs,startYs = getHarrisCorners(graay,bboxs)
    #             cv2.rectangle(img, (faces[0], faces[1]), (faces[2], faces[3]), (255, 0, 0), 2)
    #             cv2.imshow('image', img)
    #             TRACK = False
    #             enterr= True
    #             frame = imagee
    #     if enterr:
    #         frame_startt = time.time()
    #         newXs, newYs = featuresTranslation(startXs, startYs, frame, imagee)
    #         Xs, Ys ,bboxsn = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs)
    #         print("frame time: "+str(time.time() - frame_startt))
            
    #         startXs = Xs
    #         startYs = Ys

    #         if Xs.shape[0] < 15:
    #             try:
    #                 startXs,startYs = getHarrisCorners(cv2.cvtColor(imagee,cv2.COLOR_RGB2GRAY),bboxsn)
    #             except:
    #                 TRACK = True
    #                 enterr = False
            
            
    #         cv2.rectangle(img, (int(bboxsn[0][0]), int(bboxsn[0][1])), (int(bboxsn[3][0]), int(bboxsn[3][1])), (255, 0, 0), 2)
    #         cv2.imshow('img', img)
    #         bboxs = bboxsn
    #         frame = imagee

    #     # Stop if escape key is pressed
    #     k = cv2.waitKey(1)
    #     if k%256 == 27:
    #         break

    #     if k%256 == 32:
    #         TRACK = True
    #         enterr = False
            
    # # Release the VideoCapture object
    # cap.release()