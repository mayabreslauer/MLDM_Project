import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB, GaussianNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.svm import SVC
# from imblearn.over_sampling import SMOTE

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import Callback

import math
import os
import time
import sys
import subprocess
import warnings
from itertools import product
from typing import List
import IPython.display as display
from FeatureExtraction import FeatureExtraction
from PreProcessing import PreProcessing
from Utilities import Utilities
from SpiderDataExtraction import SpiderDataExtraction
class FEParameter:
    def __init__(self, name:str, min:float=-999999, max:float=999999):
        self.name = name
        self.min = min
        self.max = max
