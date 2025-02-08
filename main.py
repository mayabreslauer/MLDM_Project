import pandas as pd
from ML_Utilities import ML_Utilities
from FeatureExtraction import FeatureExtraction
from PreProcessing import PreProcessing
from Traditional_ML import Traditional_ML
from Utilities import Utilities
from SpiderDataExtraction import SpiderDataExtraction
from FeatureSelection import FeatureSelection
from FEParameter import FEParameter
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt  # Importing matplotlib's pyplot module
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    From_Feature_Selection=False
    From_classifiar=False
    part_sort=True
    part_segment=True
    part_clean=True
    part_extraction=True
    Analysis_Type='SB'
    DATABASE = 'Spider'
    data_path = r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\שנה ה סיסמסטר א\נושאים מתקדמים בעיבוד אותות פזיולוגיים\פרויקט\בסיס נתונים\spider-video-clips'
    dataframe_path = f'{data_path}_Data\Dataframes'
    #________________________________________________________SortData_______________________________________________________
    if From_classifiar is False:
        if From_Feature_Selection is False:
            if part_sort:
                sde = SpiderDataExtraction(data_path)
                # sde.download_data()
                sde.sort_data(Analysis_Type)
                Utilities.save_dataframe(sde.sorted_DATA,dataframe_path,f'{Analysis_Type}_Sorted_DATA')

            # # ________________________________________________________PreProcessing_______________________________________________________
            if part_segment:
                sde = SpiderDataExtraction(data_path)
                sde.sorted_DATA=Utilities.load_dataframe(f'{dataframe_path}/{Analysis_Type}_Sorted_DATA.csv')
                window_length_s = 30  # window length for each segment
                overlap = 0.1  # overlap for sliding window
                pp = PreProcessing(sde.sorted_DATA, sde.sampling_frequency)
                pp.segment(window_length_s,overlap)
                Utilities.save_dataframe(pp.segment_DATA,dataframe_path,f'{Analysis_Type}_segment_DATA')

            if part_clean:
                sde = SpiderDataExtraction(data_path)
                pp = PreProcessing(Utilities.load_dataframe(f'{dataframe_path}/{Analysis_Type}_Sorted_DATA.csv'),sde.sampling_frequency)
                pp.segment_DATA=Utilities.load_dataframe(f'{dataframe_path}/{Analysis_Type}_segment_DATA.csv')
                pp.clean()
                Utilities.save_dataframe(pp.preprocessed_DATA, dataframe_path, f'{Analysis_Type}_Preprocessed_DATA')
            # # # ________________________________________________________FeatureExtraction_______________________________________________________
            if part_extraction:
                sde = SpiderDataExtraction(data_path)
                pp = PreProcessing(Utilities.load_dataframe(f'{dataframe_path}/{Analysis_Type}_segment_DATA.csv'),sde.sampling_frequency)
                # window length for each segment
                pp.preprocessed_DATA=Utilities.load_dataframe(f'{dataframe_path}/{Analysis_Type}_Preprocessed_DATA.csv')
                window_length_s = 30  # window length for each segment
                pp.window_samples=int(window_length_s * sde.sampling_frequency)
                fe = FeatureExtraction(pp.preprocessed_DATA,pp.window_samples, pp.sampling_frequency)
                fe.extract(ECG=False, EDR=False, show_plot=False)
                Utilities.save_dataframe(fe.feature_extracted_DATA, dataframe_path, f'{Analysis_Type}-Full Feature Extracted RSP ECG')
        # ________________________________________________________FeatureSelection_______________________________________________________
        # fs = FeatureSelection(fe.feature_extracted_ECG)
        # fs = FeatureSelection(Utilities.load_dataframe(f'{dataframe_path}/{Analysis_Type}-Full Feature Extracted ECG BR.csv'))
        fs = FeatureSelection(Utilities.load_dataframe(f'{dataframe_path}/{Analysis_Type}-Full Feature Extracted RSP ECG.csv'))
        Complex=False
        if Complex:
            # Minimum and maximum expected HR (beats per min)
            min_HR = 30
            max_HR = 200

            ECG_Rate_Mean=FEParameter('ECG_Rate_Mean', min=min_HR, max=max_HR)

            # MinNN: The minimum of the RR intervals (Parent, 2019; Subramaniam, 2022).
            HRV_MinNN = FEParameter('HRV_MinNN', min=60000.0 / max_HR, max=60000.0 / min_HR)
            # MaxNN: The maximum of the RR intervals (Parent, 2019; Subramaniam, 2022).
            HRV_MaxNN = FEParameter('HRV_MaxNN', min=60000.0 / max_HR, max=60000.0 / min_HR)
            # MeanNN: The mean of the RR intervals.
            HRV_MeanNN = FEParameter('HRV_MeanNN', min=60000.0 / max_HR, max=60000.0 / min_HR)

            # pNN20: The proportion of RR intervals greater than 20ms, out of the total number of RR intervals.
            HRV_pNN20 = FEParameter('HRV_pNN20')
            # pNN50: The proportion of RR intervals greater than 50ms, out of the total number of RR intervals.
            HRV_pNN50 = FEParameter('HRV_pNN50')
            # A geometrical parameter of the HRV, or more specifically, the baseline width of the RR intervals distribution
            # TINN: obtained by triangular interpolation, where the error of least squares determines the triangle.
            # It is an approximation of the RR interval distribution.
            HRV_TINN = FEParameter('HRV_TINN')
            # HTI: The HRV triangular index, measuring the total number of RR intervals divided by the height of the RR intervals histogram.
            HRV_HTI = FEParameter('HRV_HTI')

            # VLF: The spectral power (W/Hz) of very low frequencies (.0033 to .04 Hz).
            HRV_VLF = FEParameter('HRV_VLF')  # hidden due to use of 0.5 Hz high-pass butterworth filter
            # LF: The spectral power (W/Hz) of low frequencies (.04 to .15 Hz).
            HRV_LF = FEParameter('HRV_LF')
            # HF: The spectral power (W/Hz) of high frequencies (.15 to .4 Hz).
            HRV_HF = FEParameter('HRV_HF')
            # LFHF: The ratio obtained by dividing the low frequency power by the high frequency power.
            HRV_LFHF = FEParameter('HRV_LFHF')

            # SDNN: The standard deviation of the RR intervals.
            # See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/ for chosen max value.
            HRV_SDNN = FEParameter('HRV_SDNN', max=200)
            # RMSSD: The square root of the mean of the squared successive differences between adjacent RR intervals.
            # # It is equivalent (although on another scale) to SD1, and therefore it is redundant to report correlations with both (Ciccone, 2017).
            # See https://help.welltory.com/en/articles/4413231-what-normal-ranges-and-measurement-standards-we-use-to-interpret-your-heart-rate-variability for chosen max value.
            HRV_RMSSD = FEParameter('HRV_RMSSD', max=200)
            # The root mean square of successive differences (RMSSD) divided by the mean of the RR intervals (MeanNN).
            cv_foldsSD = FEParameter('HRV_cv_foldsSD')
            # Shannon Entropy
            HRV_ShanEn = FEParameter('HRV_ShanEn')
            # Sample Entropy
            HRV_SampEn = FEParameter('HRV_SampEn')
            # DFA_alpha1: The monofractal detrended fluctuation analysis of the HR signal, corresponding to short-term correlations.
            HRV_DFA_alpha1 = FEParameter('HRV_DFA_alpha1')

            HRV_SDSD=FEParameter('HRV_SDSD')

            # ECG Intervals and Durations
            parameter_names = ['ECG_S_Peaks_Interval_Mean',
                               'ECG_S_Peaks_Interval_SD',
                               'ECG_T_Peaks_Interval_Mean',
                               'ECG_T_Peaks_Interval_SD',
                               'ECG_P_Duration_Mean',
                               'ECG_P_Duration_SD',
                               'ECG_T_Duration_Mean',
                               'ECG_T_Duration_SD']

            # create new objects for each parameter name
            ECG_duration_intervals = []
            for name in parameter_names:
                parameter = FEParameter(name)
                ECG_duration_intervals.append(parameter)

            # PSD
            parameter_names = ['ECG_FQ_10', 'ECG_FQ_20', 'ECG_FQ_30', 'ECG_FQ_40', 'ECG_FQ_50', 'ECG_FQ_60', 'ECG_FQ_70',
                               'ECG_FQ_80', 'ECG_FQ_90', 'ECG_FQ_100', 'ECG_FQ_110', 'ECG_FQ_120', 'ECG_FQ_130', 'ECG_FQ_140',
                               'ECG_FQ_150', 'ECG_FQ_160', 'ECG_FQ_170', 'ECG_FQ_180', 'ECG_FQ_190', 'ECG_FQ_200', 'ECG_FQ_210']
            ECG_PSD = []
            for name in parameter_names:
                parameter = FEParameter(name)
                ECG_PSD.append(parameter)

            # EDR Distance: Breathing interval measured in ms.
            # Breathing rate range is approximately 12 to 25 per minute - see https://my.clevelandclinic.org/health/articles/10881-vital-signs
            EDR_Distance = FEParameter('EDR_Distance', min=2000, max=5000)
            # HRV

            # # Intervals and Durations
            # selected_features.extend(ECG_duration_intervals)
            # # PSD
            # selected_features.extend(ECG_PSD)
            # # EDR
            # selected_features.append(EDR_Distance)
            #
            # selected_features = [ECG_Rate_Mean,HRV_MeanNN, HRV_SDNN, HRV_RMSSD,HRV_pNN20, HRV_pNN50,HRV_MaxNN,HRV_VLF,HRV_LF,HRV_HF,HRV_LFHF]

        min_HR = 30
        max_HR = 200
        HRNmean =FEParameter('HRNmean')
        HRstd = FEParameter('HRstd')
        HRNSD = FEParameter('HRNSD')
        HRNFD = FEParameter('HRNFD')
        avNN = FEParameter('avNN')
        sdNN = FEParameter('sdNN')
        rMSSD = FEParameter('rMSSD')
        PHRNN50 = FEParameter('PHRNN50')
        PHRNN20 = FEParameter('PHRNN20')
        BRNmean = FEParameter('BRNmean')
        BRstd = FEParameter('BRstd')
        BRNFD = FEParameter('BRNFD')
        BRNSD = FEParameter('BRNSD')
        BRV = FEParameter('BRV')
        BRavNN = FEParameter('BRavNN')
        # BRsdNN = FEParameter('BRsdNN')

        selected_features = [HRNmean, HRstd, HRNSD, HRNFD, avNN, sdNN, rMSSD, PHRNN50, PHRNN20,BRNmean, BRstd, BRNFD, BRNSD, BRV, BRavNN]
        #
        fs.select(selected_features)

        fs.impute()
        # fs.impute1()
        # ________________________________________________________Sequential Feature Selection_______________________________________________________

        K=6
        selected_features_table = pd.DataFrame()
        X = fs.selected_features_DATA.drop(columns=['Stress Level'])
        y = fs.selected_features_DATA['Stress Level']
    # scores_table = pd.DataFrame(columns=['K'], index=['accuracy', 'std_accuracy', 'precision', 'recall', 'f1_score_mean'])
    # for K in np.arange(3,4):
        selected_features_by_FS=False
        selected_features_by_FI=False
        if selected_features_by_FS or selected_features_by_FI:
            if selected_features_by_FS:
                selected_features=pd.read_csv(f'{dataframe_path}/{Analysis_Type}-K_{K}_data_selected_features_RSP_ECG_FS.csv')
            if selected_features_by_FI:
                selected_features=pd.read_csv(f'{dataframe_path}/{K}_selected_features_FI.csv')
            if From_classifiar:
                fs = FeatureSelection(selected_features)
            fs.selected_features_DATA=selected_features

        else:
            FS_SVM=True
            f_importance=False
            visualise=False

            # fs.feature_extracted_ECG = fs.feature_extracted_ECG.dropna(axis=1, how='any')
            # X = fs.feature_extracted_ECG.drop(columns=['Stress Level'])
            # y = fs.feature_extracted_ECG['Stress Level']


            if FS_SVM:
                # svm = SVC(kernel='poly', C=1000, degree=4, gamma=0.1, random_state=15, verbose=True)
                svm = SVC(kernel='poly', degree=2, verbose=True)
                sfs = SequentialFeatureSelector(svm, n_features_to_select=K,cv=10)
                # sfs = SequentialFeatureSelector(svm, n_features_to_select=K,cv=10,scoring='accuracy')
                sfs.fit(X,y)
                sfs.get_support()
                sfs.transform(X).shape
                selected_features = pd.DataFrame(sfs.feature_names_in_[sfs.support_], columns=[f'Iteration_{K}'])
                selected_features_table = pd.concat([selected_features_table, selected_features], axis=1)
                selected_features = np.insert(selected_features, 0, "Stress Level")
                print(K)
                fs.selected_features_DATA = fs.feature_extracted_DATA[selected_features]
                Utilities.save_dataframe(fs.selected_features_DATA, dataframe_path, f'{Analysis_Type}-K_{K}_data_selected_features_RSP_ECG_FS')

            if f_importance:
                model = DecisionTreeClassifier()
                # fit the model
                model.fit(X,y)
                # get importance
                importance = model.feature_importances_
                feature_names =X.columns.tolist()
                importance_with_names = [(feature_names[i], importance[i]) for i in range(len(importance))]

                # for i, v in enumerate(importance):
                #     print('Feature: %0d, Score: %.5f' % (i, v))
                # # plot feature importance
                # pyplot.bar([x for x in range(len(importance))], importance)
                # pyplot.show()
                # # summarize feature importance

                for i, (feature_name, score) in enumerate(importance_with_names):
                    print('Feature: %s, Score: %.5f' % (feature_name, score))

                # Plot feature importance
                pyplot.bar([name for name, _ in importance_with_names], [score for _, score in importance_with_names])
                pyplot.xticks(rotation=45)
                pyplot.show()

                # K = 3
                # top_K_indices = importance.argsort()[-K:][::-1]
                # # Get the names of top K features
                K = 6
                top_K_indices = np.argsort([score for _, score in importance_with_names])[::-1][:K]
                top_K_features = [feature_names[idx] for idx in top_K_indices]
                print_K_features = [f"Feature {feature_names[idx]}" for idx in top_K_indices]
                X_train_selected = X.iloc[:, top_K_indices]
                print("Top", K, "Features:", print_K_features)
                print("X_train shape with selected features:", X_train_selected.shape)
                # Create X_train with only the top K features
                columns_to_select = ["Stress Level"]+top_K_features
                # Train_data[f'{K}'] = fs.selected_features_ECG[columns_to_select]
                fs.selected_features_DATA=fs.selected_features_ECG[columns_to_select]
                Utilities.save_dataframe(columns_to_select, dataframe_path, f'K_{K}_selected_features_FI')

            if visualise:
                # fs.visualise(plot_type='kdeplot', single_feature='ECG_P_Duration_Mean')
                # fs.visualise(plot_type='kdeplot')
                fs.visualise(plot_type='pairplot')
        if not selected_features_by_FS and not selected_features_by_FI:
            Utilities.save_dataframe(selected_features_table, dataframe_path, f'{Analysis_Type}-K_{K}_tables_selected_features_RSP_ECG_FS')
        # Utilities.save_dataframe(selected_features, dataframe_path, f'{Analysis_Type}-Feature Selected ECG BR')
        # ________________________________________________________Traditional Machine Learning Methods_______________________________________________________

        selected_features_DATA= fs.selected_features_DATA
        # selected_features_ECG = Utilities.load_dataframe(f'{dataframe_path}/Feature Selected.csv')
        # make two datasets - binary and three-level classification
        dataset_binary = ML_Utilities.prepare(fs.selected_features_DATA, num_of_labels=2, test_size=0.30, n_splits=10, normalise=True)
        dataset_three_level = ML_Utilities.prepare(fs.selected_features_DATA, num_of_labels=3, test_size=0.30, n_splits=10, normalise=True)

        tml = Traditional_ML(dataset_binary, dataset_three_level, number_of_cores=1,data_path=data_path,K=K)

        # 'NB' for Naive Bayes
        # 'SVM' for Support Vector Machine
        # 'RF' for Random Forest
        tml.model = 'SVM'
        # tml.tune()
        tml.classify()
                # scores_table[f'{K}']=[mean_accuracy, std_accuracy, precision_mean, recall_mean, f1_score_mean]

        # print("_________________________________________________________")
        # tml.model = 'RF'
        # tml.tune()
        # tml.classify()

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
