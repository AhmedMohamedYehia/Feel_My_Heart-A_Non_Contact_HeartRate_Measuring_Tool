def padding(x):
    n = len(x)
    power = math.ceil(math.log(n,2))+5
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

def apply_fft(x):
    newx = padding(x)
    X=FFT(newx)
    return x