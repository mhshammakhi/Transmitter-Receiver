import os
import sys
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

inFile = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if inFile is None:
    inFile = 'output.bin'
data = np.fromfile(inFile, dtype=np.float32)

signal = data[0::2] + 1j * data[1::2]

f, pxx = welch(signal, fs=1.0, window='hamming', nperseg=2048,
                noverlap=1024, nfft=2048, return_onesided=False, detrend=False)
f = np.fft.fftshift(f)
pxx = np.fft.fftshift(pxx)

plt.plot(f, 10 * np.log10(pxx))
plt.xlabel('Normalized Frequency (cycles/sample)')
plt.ylabel('Power/Frequency (dB/(cycles/sample))')
plt.title('Welch Power Spectral Density Estimate')
plt.grid(True)
plt.show()
