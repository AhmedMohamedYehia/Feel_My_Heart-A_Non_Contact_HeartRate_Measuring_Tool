
import numpy as np
import matplotlib.pyplot as plt

def plotSignals(signals, label,WINDOW_TIME_SEC,FPS,hr_count):
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
    colors = ["r", "g", "b"]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(signals[:,i], colors[i])
    plt.xlabel('Time (sec)', fontsize=17)
    plt.ylabel(label, fontsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.savefig('./figs/signal/'+str(hr_count)+'.png')
    # plt.show()
def plotSignals_norm(signals, label,WINDOW_TIME_SEC,FPS,hr_count):
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
    colors = ["r", "g", "b"]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(signals[:,i], colors[i])
    plt.xlabel('Time (sec)', fontsize=17)
    plt.ylabel(label, fontsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.savefig('./figs/signalNorm/'+str(hr_count)+'.png')
    # plt.show()

def plotSpectrum(freqs, power_spectrum,hr_count):
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
    plt.savefig('./figs/powerSpectrum/'+str(hr_count)+'.png')
    # plt.show()
