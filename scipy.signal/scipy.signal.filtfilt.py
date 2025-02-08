import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

t = np.linspace(0, 1.0, 2001)
xlow = np.sin(2 * np.pi * 5 * t)
xhigh = np.sin(2 * np.pi * 250 * t)
x = xlow + xhigh

b, a = signal.butter(8, 0.125)
y = signal.filtfilt(b, a, x, padlen=150)
np.abs(y - xlow).max()
9.1086182074789912e-06

b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.

rng = np.random.default_rng()
n = 60
sig = rng.standard_normal(n)**3 + 3*rng.standard_normal(n).cumsum()

fgust = signal.filtfilt(b, a, sig, method="gust")
fpad = signal.filtfilt(b, a, sig, padlen=50)
plt.plot(sig, 'k-', label='input')
plt.plot(fgust, 'b-', linewidth=4, label='gust')
plt.plot(fpad, 'c-', linewidth=1.5, label='pad')
plt.legend(loc='best')
plt.show()