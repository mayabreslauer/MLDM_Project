import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import FEParameter
from sklearn.impute import SimpleImputer
import math
from typing import List



# Define FeatureSelection class that is used to visualise and select data
class FeatureSelection():
    def __init__(self, feature_extracted_DATA):
        self.feature_extracted_DATA = feature_extracted_DATA
        self.convert_to_numeric()

    def convert_to_numeric(self):
        # Convert string values to numeric
        def convert_value(value):
            if isinstance(value, str):
                # Strip brackets and convert to float
                return float(value.strip("[]"))
            else:
                return value

        # Apply the conversion function to each element of the DataFrame
        self.feature_extracted_DATA = self.feature_extracted_DATA.applymap(convert_value)
    def select(self, desired_features: List[FEParameter]):
        self.selected_features_DATA = self.feature_extracted_DATA[['Stress Level']]

        for feature in desired_features:
            out_of_range_count = 0
            # Sanity check: check if feature exists
            name=feature.name
            if feature.name in self.feature_extracted_DATA.columns:
                # Set value to NaN if it falls outside min and max values.
                for i, value in enumerate(self.feature_extracted_DATA[feature.name]):
                    if (value < feature.min) or (value > feature.max):
                        out_of_range_count += 1
                        self.feature_extracted_DATA.loc[i, feature.name] = np.nan
                # Add column to new selected features
                pd.options.mode.chained_assignment = None
                self.selected_features_DATA[feature.name] = self.feature_extracted_DATA[[feature.name]].copy()
                pd.options.mode.chained_assignment = 'warn'
            else:
                print(f'Error: No such feature "{name}" in extracted features')
            if out_of_range_count != 0:
                print(
                    f'Feature: {feature.name} is out of range {out_of_range_count}/{len(self.feature_extracted_feature_extracted_DATA[feature.name])} segments')

    # impute missing values in dataset with mean values of column
    def impute(self):
        # switch infs to NaNs
        pd.options.mode.chained_assignment = None
        self.selected_features_DATA.replace([np.inf, -np.inf], np.nan, inplace=True)
        pd.options.mode.chained_assignment = 'warn'
        # check for columns with only NaNs and delete if necessary
        drop_cols = [col for col in self.selected_features_DATA.columns if
                     self.selected_features_DATA[col].isnull().all()]
        if drop_cols:
            self.selected_features_DATA = self.selected_features_DATA.drop(drop_cols, axis=1)
        imp = SimpleImputer(strategy='mean')
        imp.fit(self.selected_features_DATA)
        self.selected_features_DATA = pd.DataFrame(imp.transform(self.selected_features_DATA), columns=self.selected_features_DATA.columns)

    def impute1(self):
        # switch infs to NaNs
        pd.options.mode.chained_assignment = None
        self.feature_extracted_DATA.replace([np.inf, -np.inf], np.nan, inplace=True)
        pd.options.mode.chained_assignment = 'warn'
        # check for columns with only NaNs and delete if necessary
        drop_cols = [col for col in self.feature_extracted_DATA.columns if self.feature_extracted_DATA[col].isnull().all()]
        if drop_cols:
            self.feature_extracted_DATA = self.feature_extracted_DATA.drop(drop_cols, axis=1)
        imp = SimpleImputer(strategy='mean')
        imp.fit(self.feature_extracted_DATA)
        self.feature_extracted_DATA = pd.DataFrame(imp.transform(self.feature_extracted_DATA), columns=self.feature_extracted_DATA.columns)

    def visualise(self, plot_type='pairplot', single_feature=None):
        print("Generating plot...")
        if plot_type == 'pairplot':
            sns.pairplot(data=self.selected_features_DATA, hue='Stress Level', palette=['green', 'orange', 'red'])

        elif plot_type == 'kdeplot':
            if single_feature is None:
                for i, feature in enumerate(self.selected_features_DATA):
                    fig = plt.figure(figsize=(8, 6))
                    sns.kdeplot(x=feature, data=self.selected_features_DATA, hue='Stress Level', common_norm=False,
                                warn_singular=False, palette=['green', 'orange', 'red'])
            else:
                sns.kdeplot(x=single_feature, data=self.selected_features_DATA, hue='Stress Level', common_norm=False,
                            warn_singular=False, palette=['green', 'orange', 'red'], clip=(0.0, 150))


        elif plot_type == 'kdesubplot':
            # Create a figure with subplots for each feature
            subplot_size = math.ceil(math.sqrt(len(self.selected_features_DATA.columns)))
            fig = plt.figure(figsize=(20, 8 * subplot_size))

            # Loop through each feature and add it to a subplot
            for i, feature in enumerate(self.selected_features_DATA):
                fig.add_subplot(subplot_size, subplot_size, i + 1)
                sns.kdeplot(x=feature, data=self.selected_features_DATA, hue='Stress Level', common_norm=False,
                            warn_singular=False, palette=['green', 'orange', 'red'])
            plt.show()

        else:
            print("Plot type not recognised. Please choose between pairplot, kdeplot, kdesubplot")
