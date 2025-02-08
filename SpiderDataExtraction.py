import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import os
import sys
import subprocess
from Utilities import Utilities
from scipy.signal import find_peaks
import neurokit2 as nk



class SpiderDataExtraction():
    def __init__(self, directory):
        self.directory = directory
        # self.directory = directory + 'Spider'
        self.sampling_frequency = 100
        self.sorted_DATA = pd.DataFrame(columns=['Timestamp', 'ECG', 'BR', 'EDA', 'Stress Level'])


    # sorts data from each participant, labelling each ECG recording and appends to one dataframe.
    # Following the SB approach in the study.
    def sort_data(self,Sort_Type=''):

        directory = self.directory+ '/'

        # Exclude VP70 because of noise
        sub_directories = ['VP02', 'VP03', 'VP05', 'VP06', 'VP08', 'VP09', 'VP11', 'VP12', 'VP14', 'VP15', 'VP17',
                           'VP18', 'VP20', 'VP23', 'VP24', 'VP26', 'VP27',
                           'VP29', 'VP30', 'VP32', 'VP33', 'VP35', 'VP36', 'VP38', 'VP39', 'VP41', 'VP42', 'VP44',
                           'VP45', 'VP47', 'VP48', 'VP50', 'VP51', 'VP53',
                           'VP54', 'VP56', 'VP57', 'VP59', 'VP61', 'VP62', 'VP63', 'VP64', 'VP65', 'VP66', 'VP68',
                           'VP69', 'VP71', 'VP72', 'VP73', 'VP74',
                           'VP75', 'VP76', 'VP77', 'VP78', 'VP79', 'VP80']

        # Path to Ratings file for all particpants

        subjective_ratings_file = f'{self.directory}/Subjective Ratings.txt'

        # Read in the subject ratings file (ignore arousal markers, interested in angst)
        ratings_df = (pd.read_csv(subjective_ratings_file, sep='\t', names=['Subject','Group','Session', '4', '8', '12', '16', 'NA1', 'NA2', 'NA3', 'NA4'], encoding='UTF-16')
            .drop(columns=['NA1', 'NA2', 'NA3', 'NA4'])
            .iloc[1:]
            .reset_index(drop=True)
            .astype(int)
        )

        for index, sub_directory in enumerate(sub_directories):
            Utilities.progress_bar('Sorting database', index, len(sub_directories)-1)

            # set participant data paths
            ECG_file = f'{directory}{sub_directory}/BitalinoECG.txt'
            BR_file = f'{directory}{sub_directory}/BitalinoBR.txt'
            EDA_file = f'{directory}{sub_directory}/BitalinoGSR.txt'
            triggers_file = f'{directory}{sub_directory}/Triggers.txt'

            # Get participant number
            participant_no = int(sub_directory[2:])

            # read in particpant ECG raw data file and reorder to get columns Timestamp, ECG
            raw_ECG = pd.read_csv(ECG_file, sep='\t', names=['ECG', 'Timestamp', 'NA'])
            raw_ECG = raw_ECG.drop(columns=['NA'])
            raw_ECG = raw_ECG[['Timestamp', 'ECG']]

            raw_BR = pd.read_csv(BR_file, sep='\t', names=['BR', 'Timestamp', 'NA'])
            raw_BR = raw_BR.drop(columns=['NA'])
            raw_BR = raw_BR[['Timestamp', 'BR']]

            raw_EDA = pd.read_csv(EDA_file, sep='\t', names=['EDA', 'Timestamp', 'NA'])
            raw_EDA = raw_EDA.drop(columns=['NA'])
            raw_EDA = raw_EDA[['Timestamp', 'EDA']]

            raw_df = raw_ECG.merge(raw_BR, on='Timestamp').merge(raw_EDA, on='Timestamp')
            # Read in participant trigger file
            triggers_df = pd.read_csv(triggers_file, sep='\t', names = ['Clip','On','Off'])

            # Determine stress levels by correspoding the raw ecg data with the triggers file and the Subjective Ratings file.
            # Iterate through the 16 stress clips (first clip is a demo):
            for i in range(1, 17):
                # Determine row in ratings file
                row = ratings_df.loc[ratings_df['Subject'] == participant_no].index[0]

                # find stress for the clip in the ratings file
                stress_level = ratings_df.iloc[row]['4'] if i <= 4 else ratings_df.iloc[row]['8'] if i <= 8 else ratings_df.iloc[row]['12'] if i <= 12 else ratings_df.iloc[row]['16']

                # convert stress level to Low, Medium or High (1-3)
                stress_level = 2 if (stress_level <= 2) else 3

                # Get 60 second slice of ECG data for that clip
                clip_start_time = int(triggers_df.iloc[i]['On'])
                start_index = raw_df.index[raw_df['Timestamp']>clip_start_time].tolist()[0]
                clip_df = raw_df.iloc[start_index:start_index + (self.sampling_frequency * 60)].copy(deep=False)
                clip_df['Stress Level'] = stress_level
                self.sorted_DATA = pd.concat([self.sorted_DATA, clip_df], axis=0, ignore_index=True)

            # Add the last 3 minute resting phase (stress level Low) to the data
            rest_start_time = int(triggers_df.iloc[-1]['On'])
            start_index = (raw_df['Timestamp'] > rest_start_time).idxmin() + (self.sampling_frequency * 120)
            rest_df = raw_df.iloc[start_index: start_index + (self.sampling_frequency * 180)].copy(deep=False)
            rest_df.loc[:, 'Stress Level'] = 1
            self.sorted_DATA = pd.concat([self.sorted_DATA, rest_df], axis=0, ignore_index=True)


