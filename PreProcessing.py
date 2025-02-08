import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt,freqs
from Utilities import Utilities
import matplotlib.pyplot as plt

class PreProcessing():
    def __init__(self, sorted_DATA: pd.DataFrame, sampling_frequency: int):
        self.sorted_DATA = sorted_DATA
        self.sampling_frequency = sampling_frequency
        self.segment_DATA= pd.DataFrame()
        self.preprocessed_DATA= pd.DataFrame()
        self.window_samples=0
    # segments data with overlap using rolling window, if semgnet_heartbeats true, then segment is centralised around the heartbeat (R peak)
    def segment(self, window_length_s: float, overlap: float, segment_hearbeats=False):
        # convert window_length in seconds to samples
        self.window_samples = int(window_length_s * self.sampling_frequency)
        # Calculate the step_size as the fraction of the total window samples
        step_size = int(self.window_samples * (1 - overlap))

        # Initialize starting variables
        current_index = 0
        current_stressed = self.sorted_DATA['Stress Level'][current_index]
        self.segment_DATA = pd.DataFrame(columns=['Timestamp', 'ECG', 'BR', 'EDA', 'Stress Level'])

        # faster to concatenate at the end
        preprocessed_DATA_list = []

        # get all R peaks index if required
        if segment_hearbeats:
            r_peaks = nk.ecg_peaks(self.sorted_DATA['ECG'], sampling_rate=self.sampling_frequency)

        # Loop through the entire dataframe
        while current_index < len(self.sorted_DATA['Timestamp']):
            Utilities.progress_bar('Segmenting data', current_index, len(self.sorted_DATA))
            # calculate end index in window and exit if out of bounds
            end_index = current_index + self.window_samples
            if (end_index > len(self.sorted_DATA['Timestamp'])):
                break

            # Check if the window overlaps a different label
            end_stressed = self.sorted_DATA['Stress Level'][end_index]

            # If the next window has a different label, skip to next start of next label
            if end_stressed != current_stressed:
                while (current_stressed == self.sorted_DATA['Stress Level'][current_index]):
                    current_index += 1
                current_stressed = end_stressed

            # otherwise, add segment to list of pre-processed ECG
            else:
                if segment_hearbeats:
                    # get index of next r peak
                    while not bool(r_peaks[0]['ECG_R_Peaks'][current_index]):
                        current_index += 1
                    # append segment centred on r-peak to dataframe
                    preprocessed_DATA_list.append(self.sorted_ECG.iloc[(current_index - (self.window_samples // 2)):(
                                current_index + (self.window_samples // 2))].astype('Float64'))
                    # shift the window to next non r-peak index
                    current_index += 1
                else:
                    # append segment to dataframe
                    preprocessed_DATA_list.append(self.sorted_DATA.iloc[current_index:current_index + self.window_samples].astype('Float64'))
                    # Shift the window
                    current_index += step_size

        self.segment_DATA = pd.concat(preprocessed_DATA_list, axis=0, ignore_index=True).astype('Float64')
        Utilities.progress_bar('Segmenting data', current_index, current_index)

    def create_2d(self):
        # convert the pandas DataFrame into a 2D pandas where each row has the size of window and the corresponding label (stress level)

        # Calculate the number of rows required
        num_rows = len(self.preprocessed_ECG['ECG']) // self.window_samples

        # Create an empty dataframe to hold the reshaped data
        df_reshaped = pd.DataFrame(index=range(num_rows), columns=[f"ECG {i}" for i in range(self.window_samples)])

        # Reshape the data
        for i in range(num_rows):
            start_idx = i * self.window_samples
            end_idx = (i + 1) * self.window_samples
            values = self.preprocessed_ECG['ECG'].iloc[start_idx:end_idx].values
            df_reshaped.iloc[i, :] = values

        self.preprocessed_ECG_2d = df_reshaped
        self.preprocessed_ECG_2d['Stress Level'] = self.preprocessed_ECG['Stress Level'][::self.window_samples].reset_index(drop=True)

    def clean(self):
        plot=False
        fft=False
        butter_filter=False
        # Clean each sample in the stressed and not stressed data (overwrites original data)
        # using method 'neurokit' (0.5 Hz high-pass butterworth filter (order = 5), followed by powerline filtering) but can be changed to other cleaning methods
        print("Cleaning data...")
        # Define filter order and cutoff frequency
        self.preprocessed_DATA['Timestamp'] = self.segment_DATA[['Timestamp']]
        self.preprocessed_DATA['ECG'] = pd.Series(nk.ecg_clean(self.segment_DATA['ECG'], self.sampling_frequency, method='neurokit')).astype('Float64')
        self.preprocessed_DATA['BR'] = pd.Series(nk.rsp_clean(self.segment_DATA['BR'], self.sampling_frequency, method='BioSPPy')).astype('Float64')
        self.preprocessed_DATA['Stress Level'] = self.segment_DATA[['Stress Level']]

        if butter_filter:
            # ___________________________________________________________________________Cleaning_ECG_______________________________________________________________________________________________
            lowcut = 0.5  # Lower cutoff frequency (Hz)
            order = 5  # Filter order - changed to 5
            # Design Butterworth filter
            b, a = butter(order, lowcut, btype='highpass', analog=False, fs=self.sampling_frequency)
            # # Filter ECG signal
            filtered_ecg = filtfilt(b, a, self.segment_DATA['ECG'])
        if plot:
            plt.plot(self.segment_DATA['ECG'][0:6000], label='Original Signal')
            plt.plot(self.preprocessed_DATA['ECG'][0:6000], label='Package Filtered Signal')
            plt.plot(filtered_ecg, label='Butter Filtered Signal')
            plt.xlabel('Time')
            plt.ylabel('Signal Amplitude')
            plt.title('Original and Filtered Signal')
            plt.legend()
            plt.show()
            lowcut = 0.1  # Lower cutoff frequency (Hz)
            highcut = 0.4  # Higher cutoff frequency (Hz)
            order = 3  # Filter order
            # Design Butterworth filter
            b, a = butter(order, [lowcut, highcut], btype='bandpass', analog=False, fs=self.sampling_frequency)
            filtered_rsp = filtfilt(b, a, self.segment_DATA['BR'])

            # Plot the original and filtered signal
            plt.plot(self.segment_DATA['BR'][0:6000], label='Original Signal')
            plt.plot(filtered_rsp[0:6000], label='Butter Filtered Signal')
            plt.plot(self.preprocessed_DATA['BR'][0:6000], label='Package Filtered Signal')
            plt.xlabel('Time')
            plt.ylabel('Signal Amplitude')
            plt.title('Original and Filtered RSP Signal')
            plt.legend()
            plt.show()

            self.preprocessed_DATA['BR'] = pd.Series(filtered_rsp).astype('float64')

        if fft:
            # Generate sample ECG signal (you would replace this with your actual data)
            # For demonstration purposes, let's create a synthetic ECG signal with known frequency components
            fs = 100  # Sampling frequency (Hz)

            # Perform FFT on ECG signal
            fft_ecg = np.fft.fft(self.preprocessed_DATA['ECG'])
            freqs = np.fft.fftfreq(len(fft_ecg), 1 / fs)

            # Plot FFT of ECG signal
            plt.figure(figsize=(10, 6))
            plt.plot(freqs[:len(freqs) // 2], np.abs(fft_ecg[:len(fft_ecg) // 2]))  # Plot only positive frequencies
            plt.title('FFT of ECG Signal')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.show()

            fft_rsp = np.fft.fft(self.preprocessed_DATA['BR'])
            freqs = np.fft.fftfreq(len(fft_rsp), 1 / fs)

            # Plot FFT of ECG signal
            plt.figure(figsize=(10, 6))
            plt.plot(freqs[:len(freqs) // 2], np.abs(fft_rsp[:len(fft_rsp) // 2]))  # Plot only positive frequencies
            plt.title('FFT of RSP Signal')
            plt.xlim([0,1])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.show()







        # ___________________________________________________________________________Cleaning_RSP_______________________________________________________________________________________________


            # # Generate sample respiration signal (you would replace this with your actual data)
            # # For demonstration purposes, let's create a synthetic RSP signal with known frequency components
            # fs = 100  # Sampling frequency (Hz)
            # t = np.arange(0, 10, 1 / fs)  # Time vector
            # f1 = 0.2  # Frequency of respiration signal (Hz)
            # # Perform FFT on respiration signal
            # fft_rsp = np.fft.fft(self.preprocessed_DATA['BR'])
            # freqs = np.fft.fftfreq(len(fft_rsp), 1 / fs)
            # # Plot FFT of respiration signal
            # plt.figure(figsize=(10, 6))
            # plt.plot(freqs[:len(freqs) // 2], np.abs(fft_rsp[:len(fft_rsp) // 2]))  # Plot only positive frequencies
            # plt.title('FFT of Respiration Signal')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Amplitude')
            # plt.grid(True)
            # plt.show()

            # # Filter ECG signal

            # Store filtered signal back into self.preprocessed_ECG['ECG']

            # ___________________________________________________________________________Cleaning_EDA_______________________________________________________________________________________________
            # # Low-pass filter parameters
            # lowcut_lp = None  # Low cutoff frequency (None for a low-pass filter)
            # highcut_lp = 1.5  # High cutoff frequency for low-pass filter
            # order_lp = 2  # Filter order
            # # High-pass filter parameters
            # lowcut_hp = 0.05  # Low cutoff frequency for high-pass filter
            # highcut_hp = None  # High cutoff frequency (None for a high-pass filter)
            # order_hp = 2  # Filter order
            #
            # # Design Butterworth low-pass filter
            # b_lp, a_lp = butter(order_lp, highcut_lp, btype='low', analog=False, fs=self.sampling_frequency)
            # # Design Butterworth high-pass filter
            # b_hp, a_hp = butter(order_hp, lowcut_hp, btype='high', analog=False, fs=self.sampling_frequency)
            # # # Filter ECG signal
            # # Apply the low-pass filter to the EDA signal
            # filtered_eda_lp = filtfilt(b_lp, a_lp, self.preprocessed_DATA['EDA'])
            #
            # # Apply the high-pass filter to the low-pass filtered EDA signal
            # filtered_eda_hp = filtfilt(b_hp, a_hp, self.preprocessed_DATA['EDA'])
            #
            # # Store filtered signal back into self.preprocessed_ECG['ECG']
            # self.preprocessed_DATA['EDA'] = pd.Series(filtered_ecg).astype('float64')

            # self.preprocessed_ECG['ECG'] = pd.Series(nk.ecg_clean(self.preprocessed_ECG['ECG'], self.sampling_frequency, method='neurokit')).astype('Float64')
            # self.preprocessed_ECG['ECG'] = pd.Series(self.preprocessed_ECG['ECG']).astype('Float64')
            # self.preprocessed_ECG['ECG'] = pd.Series(nk.ecg_clean(self.preprocessed_ECG['ECG'], self.sampling_frequency, method='neurokit')).astype('Float64')
