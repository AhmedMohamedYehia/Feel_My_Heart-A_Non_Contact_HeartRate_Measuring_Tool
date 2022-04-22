import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA
import warnings
import random
import math

from Fourier_Transform import *

# Toggle these for different ROIs
REMOVE_EYES = True
FOREHEAD_ONLY = False
ADD_BOX_ERROR = False

USE_OUR_ICA = False
USE_OUR_FFT = False

CASCADE_PATH = "haarcascade_frontalface_default.xml"
VIDEO_DIR = "../video/"
OPEN_WEBCAM = True
DEFAULT_VIDEO = "cv_camera_sensor_stream_handler.avi"
# DEFAULT_VIDEO = "IMG_5356.mp4"
# DEFAULT_VIDEO = "android-1.mp4"

RESULTS_SAVE_DIR = "../results/"
if REMOVE_EYES:
    RESULTS_SAVE_DIR += "no_eyes/"
if FOREHEAD_ONLY:
    RESULTS_SAVE_DIR += "forehead/"

MIN_FACE_SIZE = 100

WIDTH_FRACTION = 0.6 # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1

FPS = 30
WINDOW_TIME_SEC = 1 # ???????????(sayed bey2ol bta3et el norm)
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS)) # ??????????/
MIN_HR_BPM = 60.0
MAX_HR_BMP = 210.0
MAX_HR_CHANGE = 12.0 # ne check ezay metaba2a we sha8ala walla laa ????????
SEC_PER_MIN = 60

EYE_LOWER_FRAC = 0.25
EYE_UPPER_FRAC = 0.5

BOX_ERROR_MAX = 0.5

def get_masked_roi(image, bounding_box): 
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

def get_ROI(frame, faceCascade, previous_bounding_box):

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

    # adding box error
    if bounding_box is not None:
        if ADD_BOX_ERROR:
            noise = []
            for i in range(4):
                noise.append(random.uniform(-BOX_ERROR_MAX, BOX_ERROR_MAX))
            (x, y, w, h) = bounding_box
            x1 = x + int(noise[0] * w)
            y1 = y + int(noise[1] * h)
            x2 = x + w + int(noise[2] * w)
            y2 = y + h + int(noise[3] * h)
            bounding_box = (x1, y1, x2-x1, y2-y1)

        roi = get_masked_roi(frame, bounding_box)
    return bounding_box, roi

def plotSignals(signals, label):
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
    colors = ["r", "g", "b"]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(seconds, signals[:,i], colors[i])
    plt.xlabel('Time (sec)', fontsize=17)
    plt.ylabel(label, fontsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.show()

def plotSpectrum(freqs, power_spectrum):
    idx = np.argsort(freqs)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(freqs[idx], power_spectrum[idx,i])
    plt.xlabel("Frequency (Hz)", fontsize=17)
    plt.ylabel("Power", fontsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.xlim([0.75, 4])
    plt.show()

def get_heart_rate(window, lastHR,NUMBER_OF_SECONDS_TO_WAIT,show_plots = False):
    global outlier_count
    # Normalize across the window to have zero-mean and unit variance
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std

    # Separate into three source signals using ICA
    source_signal = None
    if USE_OUR_ICA:
        pass
    else:
        ica = FastICA()
        source_signal = ica.fit_transform(normalized)

    # Find power spectrum
    power_spectrum = None
    if USE_OUR_FFT:
        s0 = apply_fft(source_signal[:, 0])
        s1 = apply_fft(source_signal[:, 1])
        s2 = apply_fft(source_signal[:, 2])

        power_spectrum = np.abs(np.vstack([s0,s1,s2]))**2
    else:
        power_spectrum = np.abs(np.fft.fft(source_signal, axis=0))**2
    freqs = np.fft.fftfreq(WINDOW_SIZE, 1.0 / FPS)

    # Find heart rate
    maxPwrSrc = np.max(power_spectrum, axis=1)
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
    validPwr = maxPwrSrc[validIdx]
    validFreqs = freqs[validIdx]
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx]*60
    if (lastHR is not None) and (abs(lastHR-hr) > MAX_HR_CHANGE):
        outlier_count += 1
        if hr > lastHR:
            hr = lastHR + MAX_HR_CHANGE
        else:
            hr = lastHR - MAX_HR_CHANGE

    if( NUMBER_OF_SECONDS_TO_WAIT == 0):
        print("------------------------------------------------------------------------")
        print("start of real reading:")
        print("-----------------------")

    print(hr)

    if show_plots:
        plotSignals(normalized, "Normalized color intensity")
        plotSignals(source_signal, "Source signal strength")
        plotSpectrum(freqs, power_spectrum)

    return hr

# Set up video and face tracking
try:
    videoFile = sys.argv[1]
except:
    videoFile = DEFAULT_VIDEO  

video = None
if OPEN_WEBCAM:
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    video = cv2.VideoCapture(VIDEO_DIR + videoFile)

faceCascade = cv2.CascadeClassifier(CASCADE_PATH)

rgb_signal = [] # Will store the average RGB color values in each frame's ROI
heart_rates = [] # Will store the heart rate calculated every 1 second
previous_bounding_box = None
NUMBER_OF_SECONDS_TO_WAIT = 15 # waits for heartrate to converge
away_count = 0
outlier_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = video.read()

    # if no frames captured break
    if not ret:
        break
    
    # get roi for current frame
    previous_bounding_box, roi = get_ROI(frame, faceCascade, previous_bounding_box)

    # if there is an roi detected get signal of avereged rgb
    if (roi is not None) and (np.size(roi) > 0):
        channels = roi.reshape(-1, roi.shape[-1])
        average_color = channels.mean(axis=0)
        rgb_signal.append(average_color)

    # perform sliding window and wait for the window size then
    # Calculate heart rate every one second (once have 30-second of data)
    if (len(rgb_signal) >= WINDOW_SIZE) and (len(rgb_signal) % np.ceil(FPS) == 0):
        windowStart = len(rgb_signal) - WINDOW_SIZE
        window = rgb_signal[windowStart : windowStart + WINDOW_SIZE]
        lastHR = heart_rates[-1] if len(heart_rates) > 0 else None
        heart_rates.append(get_heart_rate(window, lastHR,NUMBER_OF_SECONDS_TO_WAIT))
        NUMBER_OF_SECONDS_TO_WAIT -= 1

    if np.ma.is_masked(roi):
        roi = np.where(roi.mask == True, 0, roi)

    if roi is not None:
        if(len(heart_rates) > 0):
            cv2.putText(roi, str(heart_rates[-1]), ((previous_bounding_box[0]+(previous_bounding_box[2]//4)), previous_bounding_box[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.imshow('ROI', roi)
        away_count = 0
        
    else:
        frame = np.full(frame.shape, False, dtype=np.uint8)
        cv2.putText(frame, "Please recenter your face ", (90, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.imshow('ROI',frame )
        away_count +=1

    k = cv2.waitKey(30) & 0xff
    if k==32 or away_count==(FPS*3): # space press
        print("Recalibrating !")
        rgb_signal = []
    if k==27 or k==-1: # escape press
        break

if not OPEN_WEBCAM:
    print (videoFile)
    filename = RESULTS_SAVE_DIR + videoFile[0:-4]
    if ADD_BOX_ERROR:
        filename += "_" + str(BOX_ERROR_MAX)
    np.save(filename, heart_rates)

    
print (heart_rates)
print("Number of outliers: ",outlier_count)

video.release()
cv2.destroyAllWindows()