import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from Dataset import Dataset
class ML_Utilities():
    # corrects imbalanced dataset using SMOTE, and splits into n_splitted stratified shuffle split .
    def prepare(selected_features_ECG: pd.DataFrame, num_of_labels, test_size: float, n_splits: int,
                normalise=True) -> Dataset:
        # remove medium labels for binary classification
        if num_of_labels == 2:
            selected_features_ECG = selected_features_ECG[selected_features_ECG['Stress Level'] != 1]

        y = selected_features_ECG['Stress Level']
        X = selected_features_ECG.loc[:, selected_features_ECG.columns != 'Stress Level']

        # L2 normalization for each feature (if required)
        if normalise:
            X_normalized = pd.DataFrame(normalize(X.values, norm='l2', axis=0), columns=X.columns)
            # Scale the values to be between 0 and 1 with MinMaxScaler
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_normalized)
            # Replace X with the scaled values
            X = pd.DataFrame(X_scaled, columns=X.columns)

        print("Before imbalanced label correction (SMOTE):")
        low = y.value_counts()[0] / len(y) * 100
        print(f'    Low Stress in dataset: {low:.2f}%')
        if num_of_labels == 3:
            medium = y.value_counts()[1] / len(y) * 100
            print(f'    Medium Stress in dataset: {medium:.2f}%')
        high = y.value_counts()[2] / len(y) * 100
        print(f'    High Stress in dataset: {high:.2f}%')

        # Split the data into training and test sets
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=15)

        # Obtain the training and testing sets for the first fold
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # SMOTE class rebalance
            X_train, y_train = SMOTE(sampling_strategy='not majority').fit_resample(X_train, y_train)
            break

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # create data structure
        dataset = Dataset(X, y, X_train, y_train, X_test, y_test, sss, num_of_labels)
        return dataset

    def plot_confustion_matrix(num_of_labels, y_test=0, y_pred=0, cm=0,model_type='' ):

        display_labels = ['Low', 'High'] if (num_of_labels == 2) else ['Low', 'Medium', 'High']
        # Confusion Matrix
        default_font_size = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': default_font_size * 1.4})
        # plt.figure(figsize=(8, 6))  # Adjust figure size as needed

        if isinstance(cm, int):
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=display_labels, normalize='true')
        else:
            disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
            disp.plot(values_format='.2f')
            plt.title(f'Confusion Matrix for {num_of_labels} Class\n {model_type} ')
            # plt.tight_layout()  # Adjust layout to prevent cutoff
            plt.show()
        plt.rcParams.update({'font.size': default_font_size})