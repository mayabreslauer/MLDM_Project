import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt
import warnings
from Utilities import Utilities

class FeatureExtraction():
    # takes in cleaned ECG data
    def __init__(self, preprocessed_DATA: pd.DataFrame, window_samples: int, sampling_frequency: int):
        self.preprocessed_DATA = preprocessed_DATA
        self.window_samples = window_samples
        self.sampling_frequency = sampling_frequency
        self.feature_extracted_DATA = pd.DataFrame()

    def plot_segment(self, segment, ECG_processed=None, peaks=None, colors=['r', 'g', 'c', 'm', 'y', 'k']):
        # Define time array
        t = np.arange(len(segment)) / self.sampling_frequency

        # plot fft if no peaks are given
        if isinstance(peaks, type(None)):
            # Compute FFT
            fft = np.fft.fft(segment)
            freq = np.fft.fftfreq(len(segment), d=1 / self.sampling_frequency)

            # max frequency to plot
            max_freq = 100

            # Filter out negative frequencies and frequencies above max_freq
            mask = (freq >= 0) & (freq <= max_freq)
            freq = freq[mask]
            fft = fft[mask]

            # Calculate PSD
            psd = ((np.abs(fft) ** 2) / len(segment))
            psd = 10 * np.log10(psd)
            psd -= psd.max()

            # Plot raw ECG segment
            fig, ax = plt.subplots()
            ax.plot(t, segment)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')

            # Hide y-axis units
            ax.set_yticklabels([])

            # Crop the plot to the first 25% of the x-axis
            ax.set_xlim(t.max() * 0, t.max() * 0.25)

            # Add subplot
            subax = fig.add_axes([0.68, 0.65, 0.2, 0.2])
            subax.plot(freq, psd)

            # Limit x-axis to positive frequencies between 0 and max_freq
            subax.set_xlim(0, max_freq)

            # add labels
            subax.set_xlabel('Frequency (Hz)')
            subax.set_ylabel('PSD (dB)')

        # otherwise, plot peaks
        else:
            # Plot raw ECG segment
            plt.figure()
            plt.plot(t, segment)

            # Create Line2D objects for each peak type with corresponding color
            lines = [Line2D([0], [0], linestyle='--', color=colors[i]) for i in range(len(peaks))]

            # Plot peaks
            for i, peak in enumerate(peaks):
                peak_inds = np.where(ECG_processed[peak] == 1)[0]
                for ind in peak_inds:
                    plt.axvline(x=t[ind], linestyle='--', color=colors[i])

            # Add legend with the created Line2D objects and corresponding labels
            plt.legend(handles=lines, labels=peaks, loc='lower right')

            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            # Hide y-axis units
            plt.gca().set_yticklabels([])
            # Crop the plot to the first 25% of the x-axis
            plt.xlim(t.max() * 0, t.max() * 0.1)

            # Show plot
            plt.show()

    def wave_analysis(self, segment: pd.DataFrame, plot=False) -> pd.DataFrame:
        ECG_processed, info = nk.ecg_process(segment.to_numpy(dtype='float64'), sampling_rate=self.sampling_frequency, method='neurokit')

        # calculate the mean and SD of the peak intervals
        peaks = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']
        # peaks = ['ECG_R_Peaks']

        # Minimum and maximum expected HR (beats per min)
        min_HR = 30
        max_HR = 200
        min_interval = 60e6 / max_HR
        max_interval = 60e6 / min_HR

        df = pd.DataFrame()
        for peak in peaks:
            intervals = np.diff(np.where(np.array(ECG_processed[peak] == 1))) * self.sampling_frequency
            # Remove any intervals that are out of range
            intervals = intervals[(intervals >= min_interval) & (intervals <= max_interval)]

            df[f'{peak}_Interval_Mean'] = [np.mean(intervals)]
            df[f'{peak}_Interval_SD'] = [np.std(intervals)]

        if plot:
            self.plot_segment(segment, ECG_processed, peaks)

        # calculate the average length of each wave
        waves = ['P', 'R', 'T']
        # waves = ['R']

        max_duration = [120000, 120000, 200000]
        for w, wave in enumerate(waves):
            onsets = np.where(np.array(ECG_processed[f'ECG_{wave}_Onsets'] == 1))[0]
            offsets = np.where(np.array(ECG_processed[f'ECG_{wave}_Offsets'] == 1))[0]
            # find index of first element in offsets that is >= first element in onsets
            idx_offset = np.where(offsets >= onsets[0])[0][0]
            # find size of smallest array
            duration_size = min(onsets.size, offsets.size)
            # slice offsets array to start at same index as onset
            offsets = offsets[idx_offset:duration_size]
            # set onset to same length
            onsets = onsets[:duration_size]
            # calculate durations taking into account missed onset detection
            durations = []
            # iterate over elements of both arrays
            i = 0
            j = 0
            while i < len(offsets) and j < len(onsets):
                diff = offsets[i] - onsets[j]
                if diff < 0:
                    i += 1
                else:
                    durations.append(diff)
                    i += 1
                    j += 1
            durations = np.array(durations * self.sampling_frequency)
            # Remove any intervals that are out of range
            durations = durations[(durations <= max_duration[w])]

            duration_mean = np.mean(durations)
            duration_SD = np.std(durations)
            df[f'ECG_{wave}_Duration_Mean'] = duration_mean
            df[f'ECG_{wave}_Duration_SD'] = duration_SD

        wave_onsets_offsets = []
        for wave in waves:
            wave_onsets_offsets.append(f'ECG_{wave}_Onsets')
            wave_onsets_offsets.append(f'ECG_{wave}_Offsets')

        if plot:
            self.plot_segment(segment, ECG_processed, wave_onsets_offsets, colors=['r', 'r', 'g', 'g', 'b', 'b'])

        return df

    def calc_PSD(self, segment: pd.DataFrame) -> pd.DataFrame:
        # Sum the power across 10 Hz bands from 0 to 200 Hz
        # Compute the power spectrum using a Fast Fourier Transform
        PSD = nk.signal_psd(segment.to_list(), sampling_rate=self.sampling_frequency, method="welch", min_frequency=0.5,
                            max_frequency=200)
        # Create an empty list to store the binned power values
        binned_power = []
        # Set the initial frequency and bin range values
        frequency = 0
        bin_range = 10
        nyquist_frequency = self.sampling_frequency // 2

        # Loop through the frequency ranges of 10Hz
        while bin_range <= nyquist_frequency:
            # Initialize the total power for the current bin
            total_power = 0

            # Loop through the rows of the original dataframe
            for index, row in PSD.iterrows():
                # Check if the frequency falls within the current bin range
                if row['Frequency'] >= frequency and row['Frequency'] < bin_range:
                    # Add the power value to the total power
                    total_power += row['Power']

            # Calculate the logarithm of the total power for the current bin and append it to the binned_power list
            if total_power > 0:
                binned_power.append(np.log10(total_power))
            else:
                binned_power.append(-np.inf)

            # Increment the frequency and bin range values for the next iteration
            frequency += 10
            bin_range += 10

        # Create a new dataframe with the binned power values and the frequency ranges as the index
        binned_PSD = pd.DataFrame({'Power': binned_power})
        binned_PSD['Frequency Band'] = list(range(10, nyquist_frequency + 1, 10))
        # Convert to columns
        ECG_Frequencies = pd.DataFrame(columns=[f"ECG_FQ_{i}" for i in range(10, nyquist_frequency + 1, 10)])
        for i, column in enumerate(ECG_Frequencies.columns):
            ECG_Frequencies[column] = [binned_PSD.iloc[i]['Power']]

        return ECG_Frequencies

    def calc_collective_ECG_features(self) -> pd.DataFrame:
        print("Extracting Collective Features...")
        warnings.filterwarnings('ignore')  # temporarily supress warnings
        # automated pipeline for preprocessing an ECG signal
        # ECG_processed, info = nk.ecg_process(self.preprocessed_ECG['ECG'].to_numpy(dtype='float64'), sampling_rate=self.sampling_frequency, method='neurokit')
        # Define filter order and cutoff frequency
        N = 5  # Example filter order
        Wn = 0.5  #  0.5 HZ
        # Design Butterworth filter
        b, a = butter(N, Wn, btype='hp', analog=False, fs=self.sampling_frequency)
        # # Filter ECG signal
        filtered_ecg = filtfilt(b, a, self.preprocessed_ECG['ECG'])
        # Store filtered signal back into self.preprocessed_ECG['ECG']
        ECG_processed = pd.Series(filtered_ecg).astype('float64')

        events = np.arange(self.window_samples, self.preprocessed_ECG.shape[0], self.window_samples)
        epochs = nk.epochs_create(ECG_processed, events=events, sampling_rate=self.sampling_frequency)
        # calculate ECG Features such as pqrstu intevals etc.
        ECG_events = nk.ecg_analyze(epochs, sampling_rate=self.sampling_frequency, method='event-related')
        warnings.filterwarnings('default')
        return ECG_events

    def calc_HRV_features(self, r_peaks_df, segment, show_plot=False):
        np.seterr(divide="ignore", invalid="ignore")
        # skip segment if insufficient peaks are detected (otherwise will cause NK error)
        if int(r_peaks_df[r_peaks_df == 1].sum().iloc[0]) < 4:
            return

        # Extract HRV features from R-R peaks, see https://neuropsychology.github.io/NeuroKit/functions/hrv.html
        # compute HRV - time, frequency and nonlinear indices.
        warnings.filterwarnings('ignore')  # temporarily supress warnings
        HRV_time = nk.hrv_time(r_peaks_df, sampling_rate=self.sampling_frequency, show=show_plot)
        HRV_frequency = nk.hrv_frequency(r_peaks_df, sampling_rate=self.sampling_frequency, show=show_plot)
        warnings.filterwarnings('default')

        # compute Shannon Entropy (SE) using signal symbolization and discretization
        # see https://neuropsychology.github.io/NeuroKit/functions/complexity.html#entropy-shannon
        SE = nk.entropy_shannon(segment, symbolize='A')[0]
        HRV_ShanEn = pd.DataFrame([SE], columns=['HRV_ShanEn'])
        # concat to feature dataframe
        return HRV_time, HRV_frequency, HRV_ShanEn

    def calc_EDR(self, r_peaks_df, segment, show_plot) -> pd.DataFrame:
        # Get ECG Derived Respiration (EDR) and add to the data
        warnings.filterwarnings('ignore')  # temporarily supress warnings
        ecg_rate = nk.signal_rate(r_peaks_df, sampling_rate=self.sampling_frequency, desired_length=len(r_peaks_df))
        EDR_sample = nk.ecg_rsp(ecg_rate, sampling_rate=self.sampling_frequency)
        if show_plot:
            nk.signal_plot(segment)
            nk.signal_plot(EDR_sample)
        EDR_Distances = nk.signal_findpeaks(EDR_sample)["Distance"]
        EDR_Distance = pd.DataFrame([np.average(EDR_Distances)], columns=['EDR_Distance'])
        diff = np.diff(EDR_Distances)
        diff_squared = diff ** 2
        mean_diff_squared = np.mean(diff_squared)
        rmssd = np.sqrt(mean_diff_squared)
        EDR_RMSSD = pd.Series(rmssd)
        warnings.filterwarnings('default')

        return EDR_Distance, EDR_RMSSD

    # ______________________________________________________________________________________Functions_Statistices ECG___________________________________________________
    def normalize_signal(self, segment: pd.Series) -> pd.Series:
        mean = np.mean(segment)
        std = np.std(segment)
        normalized_signal = (segment - mean) / std
        return normalized_signal

    def calculate_Nmean(self, segment: pd.Series) -> float:
        N = len(segment)
        sum_xn = np.sum(segment)
        return sum_xn / N

    def calculate_std(self, signal: pd.Series) -> float:
        N = len(signal)
        mean = np.mean(signal)
        squared_diff = np.sum((signal - mean) ** 2)
        return np.sqrt(squared_diff / (N - 1))

    def calculate_mean_abs_diff(self, normalized_signal: pd.Series) -> float:
        diff = np.diff(normalized_signal)
        abs_diff = np.abs(diff)
        return np.mean(abs_diff)

    def calculate_NFD(self, signal: pd.Series) -> float:
        normalized_signal = (signal - np.mean(signal)) / np.std(signal)
        diff = np.diff(normalized_signal)
        abs_diff = np.abs(diff)
        return np.mean(abs_diff)

    def calculate_NSD(self, signal: pd.Series) -> float:
        normalized_signal = (signal - np.mean(signal)) / np.std(signal)
        diff = np.diff(normalized_signal, 2)  # Second differences
        abs_diff = np.abs(diff)
        return np.mean(abs_diff)
    # ______________________________________________________________________________________End_Functions_Statistices ECG___________________________________________________


    # Main method to extracts features from ECG using neurokit
    def extract(self, ECG: bool = False,RSP: bool = False,EDA: bool = False ,HRV_Complex: bool = False,EDR: bool = False,show_plot: bool = False):

        # collective epoch analysis
        #     ECG_events = self.calc_collective_ECG_features()
        self.window_samples=6000
        # individual epoch analysis
        sample_index = 0
        epoch_index = 0
        while sample_index < (len(self.preprocessed_DATA['Timestamp']) - self.window_samples):
            Utilities.progress_bar('Extracting Individual Features', sample_index, len(self.preprocessed_DATA['Timestamp']))
            # get segment ECG and stress level from dataframe
            segment = self.preprocessed_DATA.iloc[sample_index:sample_index + self.window_samples][['ECG', 'BR']]
            # segment = self.preprocessed_DATA.iloc[sample_index:sample_index + self.window_samples][['ECG', 'BR', 'EDA']]
            stress_level = self.preprocessed_DATA.iloc[sample_index]['Stress Level']
            features = pd.DataFrame({'Stress Level': [stress_level]})
            # extract R-R peaks
            r_peaks_df = nk.ecg_peaks(segment, sampling_rate=self.sampling_frequency, correct_artifacts=True)[0]
            sample_index += self.window_samples

            # ______________________________________________________________________________________Main___________________________________________________

            # ______________________________________________________________________________________ECG___________________________________________________

            try:
                ecg_signals, info = nk.ecg_process(segment['ECG'], sampling_rate=self.sampling_frequency)
                HRV=nk.ecg_intervalrelated(ecg_signals)
            except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                print("ECG Skipping data due to error:", e)
                continue

            # Assuming ecg_signals and info have been defined as before

            # Plot the cleaned ECG signal
            # plt.figure(figsize=(15, 5))
            # plt.plot(ecg_signals['ECG_Clean'], label='Cleaned ECG Signal')
            #
            # # Use the 'ECG_R_Peaks' from the 'info' dictionary to get R-peaks locations
            # r_peaks_indices = info['ECG_R_Peaks']
            #
            # # Plot R-peaks on the ECG signal
            # plt.scatter(r_peaks_indices, ecg_signals['ECG_Clean'].iloc[r_peaks_indices], color='red', label='R-peaks')
            #
            # plt.xlabel('Samples')
            # plt.ylabel('Amplitude')
            # plt.title('ECG Signal and Detected R-peaks')
            # plt.legend()
            # plt.show()


            HRNmean = pd.DataFrame({'HRNmean': [HRV['ECG_Rate_Mean'][0]]})
            HRstd = pd.DataFrame({'HRstd': [self.calculate_std(segment['ECG'])]})
            HRNFD = pd.DataFrame({'HRNFD': [self.calculate_NFD(segment['ECG'])]})
            HRNSD = pd.DataFrame({'HRNSD': [self.calculate_NSD(segment['ECG'])]})


            # HRV
            # avNN=HRV['HRV_MeanNN']
            avNN = pd.DataFrame({'avNN': [HRV['HRV_MeanNN'][0][0][0]]})
            # sdNN=HRV['HRV_SDNN']
            sdNN = pd.DataFrame({'sdNN': [HRV['HRV_SDNN'][0][0][0]]})
            # rMSSD=HRV['HRV_RMSSD']
            rMSSD = pd.DataFrame({'rMSSD': [HRV['HRV_RMSSD'][0][0][0]]})
            # PHRNN50=HRV['HRV_pNN50']
            PHRNN50 = pd.DataFrame({'PHRNN50': [HRV['HRV_pNN50'][0][0][0]]})
            # PHRNN20=HRV['HRV_pNN20']
            PHRNN20 = pd.DataFrame({'PHRNN20': [HRV['HRV_pNN20'][0][0][0]]})

            # HRV_time, HRV_frequency, HRV_ShanEn = self.calc_HRV_features(r_peaks_df, segment, show_plot=show_plot)
            # features = pd.concat([features, HRV_time, HRV_frequency, HRV_ShanEn], axis=1)
            features = pd.concat([features,HRNmean, HRstd, HRNSD,HRNFD,avNN,sdNN,rMSSD,PHRNN50,PHRNN20], axis=1)
            # ______________________________________________________________________________________RSP___________________________________________________
            try:
                rsp_signals, info = nk.rsp_process(segment['BR'], sampling_rate=self.sampling_frequency)
                BRV_DATA = nk.rsp_intervalrelated(rsp_signals)
                BRV_DATA2 = nk.rsp_rrv(rsp_signals, show=True)
            except ValueError as e:
                print("RPA Skipping data due to error:", e)
                continue

            # plt.plot(segment['BR'])
            # plt.show()
            # Process the RSP signal
            # rsp_signals, info = nk.rsp_process(segment['BR'], sampling_rate=self.sampling_frequency)
            # # Plot the cleaned RSP signal
            # plt.figure(figsize=(15, 5))
            # plt.plot(rsp_signals['RSP_Clean'], label='Cleaned RSP Signal')
            # # Use the 'RSP_Peaks' from the 'info' dictionary to get inspiratory peaks locations
            # # In the context of RSP, we are typically interested in inspiratory peaks
            # inspiratory_peaks_indices = info['RSP_Peaks']
            # # Plot inspiratory peaks on the RSP signal
            # plt.scatter(inspiratory_peaks_indices, rsp_signals['RSP_Clean'].iloc[inspiratory_peaks_indices],
            #             color='orange', label='Inspiratory Peaks')
            # plt.xlabel('Samples')
            # plt.ylabel('Amplitude')
            # plt.title('RSP Signal and Detected Inspiratory Peaks')
            # plt.legend()
            # plt.show()

            BRNmean = pd.DataFrame({'BRNmean': [BRV_DATA['RSP_Rate_Mean'][0]]})
            BRstd = pd.DataFrame({'BRstd': [self.calculate_std(segment['BR'])]})
            BRNFD = pd.DataFrame({'BRNFD': [self.calculate_NFD(segment['BR'])]})
            BRNSD = pd.DataFrame({'BRNSD': [self.calculate_NSD(segment['BR'])]})
            # #

            # BRV=BRV_DATA2['RRV_MeanBB']
            BRV = pd.DataFrame({'BRV': [BRV_DATA2['RRV_MeanBB'][0]]})
            # BRavNN=BRV_DATA['RAV_Mean']
            BRavNN = pd.DataFrame({'BRavNN': [BRV_DATA['RAV_Mean'][0]]})
            # BRsdNN=BRV_DATA['RRV_SD1']
            BRsdNN = pd.DataFrame({'BRavNN': [BRV_DATA['RRV_SD1'][0]]})

            features = pd.concat([features, BRNmean, BRstd, BRNFD,BRNSD,BRV,BRavNN,BRsdNN], axis=1)
            self.feature_extracted_DATA = pd.concat([self.feature_extracted_DATA, features], axis=0,ignore_index=True)

            # ______________________________________________________________________________________EDA___________________________________________________
            # # Low-pass filter parameters
            # lowcut_lp = None  # Low cutoff frequency (None for a low-pass filter)
            # highcut_lp = 1.5  # High cutoff frequency for low-pass filter
            # order_lp = 2  # Filter order
            # # Design Butterworth low-pass filter
            # b_lp, a_lp = butter(order_lp, highcut_lp, btype='low', analog=False, fs=self.sampling_frequency)
            # # # Filter ECG signal
            # # Apply the low-pass filter to the EDA signal
            # filtered_eda_lp = filtfilt(b_lp, a_lp, segment['EDA'])
            #
            # EDANmean= self.calculate_Nmean(filtered_eda_lp)
            # EDAstd=self.calculate_std(filtered_eda_lp)
            # EDANFD=self.calculate_NFD(filtered_eda_lp)
            # EDANSD=self.calculate_NSD(filtered_eda_lp)
            # # #
            # # High-pass filter parameters
            # lowcut_hp = 0.05  # Low cutoff frequency for high-pass filter
            # highcut_hp = None  # High cutoff frequency (None for a high-pass filter)
            # order_hp = 2  # Filter order
            #
            # # Design Butterworth high-pass filter
            # b_hp, a_hp = butter(order_hp, lowcut_hp, btype='high', analog=False, fs=self.sampling_frequency)
            # # Apply the high-pass filter to the low-pass filtered EDA signal
            # filtered_eda_hp = filtfilt(b_hp, a_hp, self.preprocessed_DATA['EDA'])
            # # nOR
            # # mmOR
            # # mdOR
            #
            # features = pd.concat([features, EDANmean, EDAstd, EDANFD,EDANSD], axis=1)
            # ______________________________________________________________________________________HRV_Complex___________________________________________________
            if HRV_Complex:
                warnings.filterwarnings('ignore')  # temporarily supress warnings

                if show_plot:
                    # nk.ecg_plot(ECG_processed_segment)
                    self.plot_segment(segment)
                try:

                    # Calculate ECG_HRV - different heart rate variability metrices.
                    HRV_intervals = nk.ecg_intervalrelated(segment['ECG'], sampling_rate=self.sampling_frequency)

                    # calculate waveform intervals (PQRSTU)
                    ECG_intervals = self.wave_analysis(segment['ECG'], show_plot)

                    # get the binned power spectrum frequencies from the ECG segment

                    ECG_frequencies = self.calc_PSD(segment['ECG'])

                    # add ECG_event to dataframe (ECG Features such as pqrstu intevals etc.))
                    # ECG_event = ECG_events.iloc[epoch_index].to_frame().transpose().reset_index()
                    # ECG_event = ECG_event.drop(['index', 'Label', 'Event_Onset'], axis=1)
                    # epoch_index += 1

                    # concat to dataframe
                    features = pd.concat([features, HRV_intervals, ECG_intervals,ECG_frequencies], axis=1)
                except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                    print("Skipping data due to error:", e)
                    continue

            # If not all ECG features desired, obtain HRV and Shannon Entropy
            else:
                # skip segments that do not yield HRV features
                try:
                    HRV_time, HRV_frequency, HRV_ShanEn = self.calc_HRV_features(r_peaks_df, segment, show_plot=show_plot)
                    features = pd.concat([features, HRV_time, HRV_frequency, HRV_ShanEn], axis=1)

                except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                    print("Skipping data due to error:", e)
                    continue
                # concat to feature dataframe


            # ______________________________________________________________________________________EDA___________________________________________________
            try:
                if EDR:
                    EDR_Distance, EDR_RMSSD = self.calc_EDR(r_peaks_df, segment, show_plot)

                # concat to dataframe
                features = pd.concat([features, EDR_Distance, EDR_RMSSD], axis=1)

            # concat features to main dataframe
            except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                print("Skipping data due to error:", e)
                continue
            # self.feature_extracted_DATA = pd.concat([self.feature_extracted_DATA, features], axis=0,ignore_index=True)


        Utilities.progress_bar('Extracting Neurokit Features', sample_index, sample_index)