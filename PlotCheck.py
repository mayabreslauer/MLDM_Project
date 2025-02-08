import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


ecg_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 1.5 * t)

# Detect R peaks
r_peaks, _ = find_peaks(ecg_signal, height=0.5, distance=200)

# Plot the ECG signal with R peaks
plt.figure(figsize=(10, 6))
plt.plot(ecg_signal, label='ECG signal')
plt.plot(r_peaks, ecg_signal[r_peaks], 'ro', label='R peaks')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.title('ECG Signal with R Peaks')
plt.legend()
plt.grid(True)
plt.show()


# Plot segments around R peaks
plt.figure(figsize=(10, 6))
for peak in r_peaks:
    segment_start = max(0, peak - 100)
    segment_end = min(len(ecg_signal), peak + 100)
    segment_time = t[segment_start:segment_end] - t[peak]
    segment_amplitude = ecg_signal[segment_start:segment_end]
    plt.plot(segment_time, segment_amplitude, label=f'R peak at t={t[peak]:.2f}s')
plt.xlabel('Time (s) relative to R peak')
plt.ylabel('Amplitude')
plt.title('Segments around R Peaks')
plt.legend()
plt.grid(True)
plt.show()
# ___________________________________2 class comparision____________________________
selected_features = selected_features[selected_features['Stress Level'].isin([0, 2])]  # Filter DataFrame for stress level classes 0 and 2
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your DataFrame is named df
sns.kdeplot(data=selected_features, x='HRNFD', hue='Stress Level', fill=True, common_norm=False)
plt.title('Density Plot of Stress Level Classes')
plt.xlabel('HRNFD')
plt.ylabel('Density')
plt.show()