import math
import numpy as np
def padding(x):
    n = len(x)
    power = math.ceil(math.log(n,2))+3
    padding=(2**power) - n
    padded = np.pad(x, (0, padding), 'constant')
    return padded



def FFT(x):
    N = len(x)
    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X

def apply_fft(signal):
    newsignal = padding(signal)
    sig_fft=FFT(newsignal)
    return sig_fft

def get_heart_rate(source_signal, lastHR,NUMBER_OF_SECONDS_TO_WAIT,USE_OUR_FFT,REMOVE_OUTLIERS,WINDOW_SIZE,FPS,MIN_HR_BPM,SEC_PER_MIN,MAX_HR_BMP,MAX_HR_CHANGE,outlier_count,show_plots = False):
    
    # Find power spectrum
    power_spectrum = None
    freqs = None
    if USE_OUR_FFT:
        s0 = apply_fft(source_signal[:, 0])
        s1 = apply_fft(source_signal[:, 1])
        s2 = apply_fft(source_signal[:, 2])
        a1 = np.array((s0,s1,s2)).T
        power_spectrum = np.abs(a1)**2
        freqs = np.fft.fftfreq(a1.shape[0], 1.0 / FPS)
    # ---------------------
    else:
        power_spectrum = np.abs(np.fft.fft(source_signal, axis=0))**2
        freqs = np.fft.fftfreq(WINDOW_SIZE, 1.0 / FPS)
    # ---------------------------------------
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

    # print(hr)
    return hr, source_signal, freqs, power_spectrum, outlier_count