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

def get_heart_rate(source_signal,WINDOW_SIZE=60,FPS=25,MIN_HR_BPM =60,SEC_PER_MIN =60,MAX_HR_BMP =180,show_plots = False):
    
    # Find power spectrum
    power_spectrum = None
    freqs = None
    
    s0 = apply_fft(source_signal[:, 0])
    s1 = apply_fft(source_signal[:, 1])
    s2 = apply_fft(source_signal[:, 2])
    a1 = np.array((s0,s1,s2)).T
    power_spectrum = np.abs(a1)**2
    freqs = np.fft.fftfreq(a1.shape[0], 1.0 / FPS)
    
    # Find heart rate
    maxPwrSrc = np.max(power_spectrum, axis=1)
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
    validPwr = maxPwrSrc[validIdx]
    validFreqs = freqs[validIdx]
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx]*60

    # print(hr)
    return hr