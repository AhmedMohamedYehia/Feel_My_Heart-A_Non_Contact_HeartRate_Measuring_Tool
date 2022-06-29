import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA

import ctypes
from Fourier_Transform import *
from Jade_Algo import *
from ROI import *
from Live_Plots import *
from FaceDetection import CascadeClassifier, ViolaAndJones,WeakClassifier,HaarFeature,Rectangle
from FaceTracking import getHarrisCorners , featuresTranslation , applyGeometricTransformation
ff = 0 
# Toggle between recorded video and to open webcam
OPEN_WEBCAM = True
SHOW_WHOLE_IMAGE = True
FULL_WINDOW = True

# Toggle these for different ROIs
REMOVE_EYES = True
FOREHEAD_ONLY = False

# Toggle txhese to use built from scratch algorithem or Scikit-learn algorithems
USE_OUR_ICA = False
USE_OUR_FFT = True
USE_OUR_FACE_DETECTION = True

REMOVE_OUTLIERS = True 
DETREND = False

show_plots = False
# show_plots = False

# Change Recorded video dir.

DEFAULT_VIDEO = "cv_camera_sensor_stream_handler.avi"


# Whether to output text file or np file
WRITE_HR_txt = True


CASCADE_PATH = "haarcascade_frontalface_default.xml"
# VIDEO_DIR = "D:/Uni/GP/Dataset/harun/harun_resting/"

# VIDEO_DIR = "D:/Uni/GP/Dataset/id1/alex/alex_resting/"
# VIDEO_DIR = "D:/Uni/GP/Dataset/cpi/cpi_gym/"
VIDEO_DIR = "../../video/"
RESULTS_SAVE_DIR = "../Results/"

MIN_FACE_SIZE = 100

WIDTH_FRACTION = 0.6 # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1

FPS = 25
WINDOW_TIME_SEC = 6 
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))

MIN_HR_BPM = 60.0
MAX_HR_BMP = 150.0
if ff:
    MIN_HR_BPM = 120.0
    MAX_HR_BMP = 160.0
MAX_HR_CHANGE = 12.0 
SEC_PER_MIN = 60

EYE_LOWER_FRAC = 0.25
EYE_UPPER_FRAC = 0.5


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


rgb_signal = [] # Will store the average RGB color values in each frame's ROI
heart_rates = [] # Will store the heart rate calculated every 1 second
previous_bounding_box = None
NUMBER_OF_SECONDS_TO_WAIT = 15 # waits for heartrate to converge
away_count = 0
outlier_count = 0
hr_count = 0

# Get the window size and calculate the center
user32 = ctypes.windll.user32
win_x, win_y = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)] 
win_cnt_x, win_cnt_y = [user32.GetSystemMetrics(0)/2, user32.GetSystemMetrics(1)/2] 

video = None
if OPEN_WEBCAM:
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    video = cv2.VideoCapture(VIDEO_DIR + videoFile)

while True:

    ret, frame = video.read()
    frame = np.full(frame.shape, False, dtype=np.uint8)
    cv2.putText(frame, "Welcome", (260, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    cv2.putText(frame, "Press Enter to start", (150, 260), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    cv2.putText(frame, "measuring your heart rate.", (110, 290), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

    if FULL_WINDOW:
        frame = cv2.resize(frame,(win_x, win_y))
    
    cv2.imshow('Feel My Heart',frame )
    k = cv2.waitKey(30) & 0xff
    # print(k)
    if k==13 or k==-1: # enter press
        break
    
    if k==27 or k==-1: # escape press
        break
mode = None
while True:

    while True:
        rgb_signal = []
        heart_rates = []
        ret, frame = video.read()
        frame = np.full(frame.shape, False, dtype=np.uint8)
        cv2.putText(frame, "Welcome", (260, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.putText(frame, "Press 1 to open webcam.", (150, 260), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.putText(frame, "Press 2 to load video.", (150, 290), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

        if FULL_WINDOW:
            frame = cv2.resize(frame,(win_x, win_y))
        
        cv2.imshow('Feel My Heart',frame )
        k = cv2.waitKey(30) & 0xff
        if k == 49:
            mode = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            break

        elif k == 50:
            mode = cv2.VideoCapture(VIDEO_DIR + videoFile)
            break

        
        if k==27 or k==-1: # escape press
            break
    
    # if escape key pressed then close program 
    if k==27 or k==-1: # escape press
        break

    faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
    while True:
        # Capture frame-by-frame
        ret, frame = mode.read()

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

            # detrending
            if DETREND:
                # print(window.shape)
                # print("shape before: ",np.asarray(window).shape)
                a0 = signal.detrend(np.asarray(window)[:,0])
                a1 = signal.detrend(np.asarray(window)[:,1])
                a2 = signal.detrend(np.asarray(window)[:,2])
                window = np.vstack([a0,a1,a2]).T
                # print("shape after: ",np.asarray(window).shape)
                window = signal.detrend(window,axis=0)

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

            hr,  source_signal, freqs, power_spectrum, outlier_count = get_heart_rate(source_signal, lastHR,NUMBER_OF_SECONDS_TO_WAIT,USE_OUR_FFT,REMOVE_OUTLIERS,WINDOW_SIZE,FPS,MIN_HR_BPM,SEC_PER_MIN,MAX_HR_BMP,MAX_HR_CHANGE,outlier_count)
            heart_rates.append(hr)
            hr_count +=1
            
            if show_plots:
                plotSignals_norm(normalized, "Normalized color intensity",WINDOW_TIME_SEC,FPS,hr_count)
                plotSignals(source_signal, "Source signal strength",WINDOW_TIME_SEC,FPS,hr_count)
                plotSpectrum(freqs, power_spectrum,hr_count)

            NUMBER_OF_SECONDS_TO_WAIT -= 1

        # mask the background as a black and the ROI with its color
        if np.ma.is_masked(roi):
            roi = np.where(roi.mask == True, 0, roi)

        # if there and ROI and a heartrate show a bounding bx arounf face with the measured heartrate
        if roi is not None:
            if SHOW_WHOLE_IMAGE:
                if(len(heart_rates) > 0):
                    cv2.putText(frame, str(int(heart_rates[-1])), ((previous_bounding_box[0]+(previous_bounding_box[2]//4)), previous_bounding_box[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                if FULL_WINDOW:
                    frame = cv2.resize(frame,(win_x, win_y))
                cv2.imshow('Feel My Heart', frame)
                away_count = 0 
            else:
                if(len(heart_rates) > 0):
                    cv2.putText(roi, str(int(heart_rates[-1])), ((previous_bounding_box[0]+(previous_bounding_box[2]//4)), previous_bounding_box[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2) 
                if FULL_WINDOW:
                    roi = cv2.resize(roi,(win_x, win_y))
                cv2.imshow('Feel My Heart', roi)
                away_count = 0 
        # if not only show the face without heartrate
        else:
            frame = np.full(frame.shape, False, dtype=np.uint8)
            # print("frame: ",frame.shape)
            cv2.putText(frame, "Please recenter your face ", (90, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            if FULL_WINDOW:
                frame = cv2.resize(frame,(win_x, win_y))
            # frame = cv2.resize(frame,(win_x, win_y))
            cv2.imshow('Feel My Heart',frame )
            away_count +=1

        # recalibrate if no face detected for 3 seconds then empty the sliding window and start calc. all over
        k = cv2.waitKey(30) & 0xff
        # print(k)
        if k==32 or away_count==(FPS*3): # space press
            print("Recalibrating !")
            rgb_signal = []
            heart_rates = []
        # if escape key pressed then close program 
        if k==8 or k==-1: # escape press
            break
        
        # if escape key pressed then close program 
        if k==27 or k==-1: # escape press
            break
    # if escape key pressed then close program 
    if k==27 or k==-1: # escape press
        break


# Save heartrate values to a text file
# print (videoFile)
filename = RESULTS_SAVE_DIR + "HR_Results.csv"
if(WRITE_HR_txt):
    file = open(filename,"w")
    file.close()
    
    a_file = open(filename, "a")
    for hr in heart_rates:
        a_file.write(str(hr)+"\n")
    a_file.close()
else:
    np.save(filename, heart_rates)



filename = RESULTS_SAVE_DIR + "Outlier_Count.csv"

file = open(filename,"w")
file.close()

a_file = open(filename, "a")
a_file.write(str(outlier_count))

    

# print (heart_rates)
# print("Number of outliers: ",outlier_count)

# mode.release()
cv2.destroyAllWindows()