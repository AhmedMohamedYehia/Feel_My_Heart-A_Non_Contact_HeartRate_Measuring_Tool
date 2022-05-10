import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA
import warnings
import random
import math


def get_masked_roi(image, bounding_box,WIDTH_FRACTION,HEIGHT_FRACTION,REMOVE_EYES,FOREHEAD_ONLY,EYE_LOWER_FRAC,EYE_UPPER_FRAC): 
    (x, y, w, h) = bounding_box
    # getting the adjusted bounding box
    bounding_box_adjusted = (int(x + ((1 - WIDTH_FRACTION) * w / 2)), int(y + ((1 - HEIGHT_FRACTION) * h / 2)),int(WIDTH_FRACTION * w), int(HEIGHT_FRACTION * h))

    (x, y, w, h) = bounding_box_adjusted
    # get the roi mask and darken background
        # start all pixels as one (to be considered)
    background_mask = np.ones(image.shape)
        # pixels outside bounding box as zeros (to be neglected)
    background_mask[int(y):int(y+h), int(x):int(x+w), :] = 0 
    
    (x, y, w, h) = bounding_box
    if REMOVE_EYES:
        # pixels in eye region as zeros (to be neglected)
        background_mask[int(y + h * EYE_LOWER_FRAC) : int(y + h * EYE_UPPER_FRAC), :] = 1
    if FOREHEAD_ONLY:
        # pixels in all regions as zeros (to be neglected) excpet forehead area
        background_mask[int(y + h * EYE_LOWER_FRAC) :, :] = 1

    roi = np.ma.array(image, mask = background_mask) # Masked array
    return roi

# Eculdian distance for each ROI
def distance(x, y):
    return sum((x[i] - y[i])**2 for i in range(len(x)))

def get_ROI(frame, faceCascade, previous_bounding_box,MIN_FACE_SIZE,WIDTH_FRACTION,HEIGHT_FRACTION,REMOVE_EYES,FOREHEAD_ONLY,EYE_LOWER_FRAC,EYE_UPPER_FRAC):

    # use face detection to get all faces in frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE), flags=cv2.CASCADE_SCALE_IMAGE)
    roi = None
    bounding_box = None

    # If only one face dectected, use it!
    if len(faces) == 1:
        bounding_box = faces[0]

    # if many faces detected, use one closest to that from previous frame
    elif len(faces) > 1:
        if previous_bounding_box is not None:
            # Find closest
            min_dist = float("inf")
            for face in faces:
                if distance(previous_bounding_box, face) < min_dist:
                    bounding_box = face
                    min_dist = distance(previous_bounding_box, face)
        else:
            # Chooses largest box by area (most likely to be true face)
            max_area = 0
            for face in faces:
                if (face[2] * face[3]) > max_area:
                    bounding_box = face
                    max_area = (face[2] * face[3])
        
    if bounding_box is not None:
        roi = get_masked_roi(frame, bounding_box,WIDTH_FRACTION,HEIGHT_FRACTION,REMOVE_EYES,FOREHEAD_ONLY,EYE_LOWER_FRAC,EYE_UPPER_FRAC)
    return bounding_box, roi
