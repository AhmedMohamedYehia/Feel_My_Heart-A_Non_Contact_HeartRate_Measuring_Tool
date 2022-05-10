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
from Jade_Algo import *
from ROI import *


# Toggle these for different ROIs
REMOVE_EYES = True
FOREHEAD_ONLY = False
ADD_BOX_ERROR = False

# Toggle these to use built from scratch algorithem or Scikit-learn algorithems
USE_OUR_ICA = False
USE_OUR_FFT = False
REMOVE_OUTLIERS = False

# Toggle between recorded video and to open webcam
OPEN_WEBCAM = True

# Change Recorded video dir.
DEFAULT_VIDEO = "cv_camera_sensor_stream_handler.avi"
# DEFAULT_VIDEO = "IMG_5356.mp4"
# DEFAULT_VIDEO = "android-1.mp4"



CASCADE_PATH = "haarcascade_frontalface_default.xml"
VIDEO_DIR = "../video/"
RESULTS_SAVE_DIR = "../results/"
if REMOVE_EYES:
    RESULTS_SAVE_DIR += "no_eyes/"
if FOREHEAD_ONLY:
    RESULTS_SAVE_DIR += "forehead/"

MIN_FACE_SIZE = 100

WIDTH_FRACTION = 0.6 # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1

FPS = 30
WINDOW_TIME_SEC = 6 
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))
MIN_HR_BPM = 45.0
MAX_HR_BMP = 180.0
MAX_HR_CHANGE = 12.0 
SEC_PER_MIN = 60

EYE_LOWER_FRAC = 0.25
EYE_UPPER_FRAC = 0.5

# def plotSignals(signals, label):
#     seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
#     colors = ["r", "g", "b"]
#     fig = plt.figure()
#     fig.patch.set_facecolor('white')
#     for i in range(3):
#         plt.plot(seconds, signals[:,i], colors[i])
#     plt.xlabel('Time (sec)', fontsize=17)
#     plt.ylabel(label, fontsize=17)
#     plt.tick_params(axis='x', labelsize=17)
#     plt.tick_params(axis='y', labelsize=17)
#     plt.show()

# def plotSpectrum(freqs, power_spectrum):
#     idx = np.argsort(freqs)
#     fig = plt.figure()
#     fig.patch.set_facecolor('white')
#     for i in range(3):
#         plt.plot(freqs[idx], power_spectrum[idx,i])
#     plt.xlabel("Frequency (Hz)", fontsize=17)
#     plt.ylabel("Power", fontsize=17)
#     plt.tick_params(axis='x', labelsize=17)
#     plt.tick_params(axis='y', labelsize=17)
#     plt.xlim([0.75, 4])
#     plt.show()

def get_heart_rate(source_signal, lastHR,NUMBER_OF_SECONDS_TO_WAIT,show_plots = False):
    global outlier_count
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
    if REMOVE_OUTLIERS:
        if (lastHR is not None) and (abs(lastHR-hr) > MAX_HR_CHANGE):
            outlier_count += 1
            hr = lastHR
            # if hr > lastHR:
            #     hr = lastHR + MAX_HR_CHANGE
            # else:
            #     hr = lastHR - MAX_HR_CHANGE

    if( NUMBER_OF_SECONDS_TO_WAIT == 0):
        print("------------------------------------------------------------------------")
        print("start of real reading:")
        print("-----------------------")
    print(hr)

    return hr, normalized, source_signal, freqs, power_spectrum

#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------          MAIN          -----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

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
    previous_bounding_box, roi = get_ROI(frame, faceCascade, previous_bounding_box,MIN_FACE_SIZE,WIDTH_FRACTION,HEIGHT_FRACTION,REMOVE_EYES,FOREHEAD_ONLY,EYE_LOWER_FRAC,EYE_UPPER_FRAC)

    # if there is an roi detected get signal of avereged rgb
    if (roi is not None) and (np.size(roi) > 0):
        rgb_signal.append(roi.reshape(-1, roi.shape[-1]).mean(axis=0))

    # perform sliding window and wait for the window size then
    # Calculate heart rate every one second (once have 30-second of data)
    if (len(rgb_signal) >= WINDOW_SIZE) and (len(rgb_signal) % FPS == 0):
        i_intial = len(rgb_signal) - WINDOW_SIZE
        window = rgb_signal[i_intial : i_intial + WINDOW_SIZE]
        lastHR = None
        if len(heart_rates) > 0:
            lastHR = heart_rates[-1] 

        # Normalize across the window to have zero-mean and unit variance
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        normalized = (window - mean) / std

        # Separate into three source signals using ICA
        source_signal = None
        if USE_OUR_ICA:
            source_signal = ICA(normalized.T).T
        else:
            ica = FastICA()
            source_signal = ica.fit_transform(normalized)

        hr, normalized, source_signal, freqs, power_spectrum = get_heart_rate(source_signal, lastHR, NUMBER_OF_SECONDS_TO_WAIT)
        heart_rates.append(hr)
        
        # if show_plots:
        #     plotSignals(normalized, "Normalized color intensity")
        #     plotSignals(source_signal, "Source signal strength")
        #     plotSpectrum(freqs, power_spectrum)

        NUMBER_OF_SECONDS_TO_WAIT -= 1

    # mask the background as a black and the ROI with its color
    if np.ma.is_masked(roi):
        roi = np.where(roi.mask == True, 0, roi)

    # if there and ROI and a heartrate show a bounding bx arounf face with the measured heartrate
    if roi is not None:
        if(len(heart_rates) > 0):
            cv2.putText(roi, str(int(heart_rates[-1])), ((previous_bounding_box[0]+(previous_bounding_box[2]//4)), previous_bounding_box[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.imshow('ROI', roi)
        away_count = 0 
    # if not only show the face without heartrate
    else:
        frame = np.full(frame.shape, False, dtype=np.uint8)
        cv2.putText(frame, "Please recenter your face ", (90, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.imshow('ROI',frame )
        away_count +=1

    # recalibrate if no face detected for 3 seconds then empty the sliding window and start calc. all over
    k = cv2.waitKey(30) & 0xff
    if k==32 or away_count==(FPS*3): # space press
        print("Recalibrating !")
        rgb_signal = []
        heart_rates = []
    # if escape key pressed then close program 
    if k==27 or k==-1: # escape press
        break

# to load video from dir. not from webcam
if not OPEN_WEBCAM:
    print (videoFile)
    filename = RESULTS_SAVE_DIR + videoFile[0:-4]
    if ADD_BOX_ERROR:
        filename += "_" + str(BOX_ERROR_MAX)
    np.save(filename, heart_rates)

    
print (heart_rates)
# print("Number of outliers: ",outlier_count)

video.release()
cv2.destroyAllWindows()