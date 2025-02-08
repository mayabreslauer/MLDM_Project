
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB, GaussianNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
import time
from Dataset import Dataset
from ML_Utilities import ML_Utilities
from Utilities import Utilities

# Define Traditional Machine Learning class, which implements different linear and non linear classification methods

class Traditional_ML():
    def __init__(self, dataset_binary: Dataset, dataset_three_level: Dataset, number_of_cores: int = 1,data_path='',K=2):
        self.dataset_binary = dataset_binary
        self.dataset_three_level = dataset_three_level
        self.n_jobs = number_of_cores
        self.model = 'RF'
        self.data_path=f'{data_path}_Data\Dataframes'
        self.iteration=K
    # main tune method
    def tune(self):
        if self.model == 'NB':
            tuning_func = lambda: self.NB_tuner()
        elif self.model == 'SVM':
            tuning_func = lambda: self.SVM_tuner()
        elif self.model == 'RF':
            tuning_func = lambda: self.RF_tuner()
        else:
            raise ValueError(f'Unsupported model: {self.model}')
        tuning_func()

    # main classification method
    def classify(self):
        if self.model == 'NB':
            tuning_func = lambda: self.NB_classifier()
        elif self.model == 'SVM':
            tuning_func = lambda: self.SVM_classifier()
        elif self.model == 'RF':
            tuning_func = lambda: self.RF_classifier()
        else:
            raise ValueError(f'Unsupported model: {self.model}')
        tuning_func()
        return tuning_func

    def evaluate(self, model, dataset: Dataset, plot_CM=True):
        # Cross-validation using Stratified ShuffleSplit
        # Initialize the accuracy and confusion matrix lists that will be averaged
        accuracies = []
        confusion_matrices = []
        thr_acc = []
        tmr_acc = []
        tlr_acc = []
        run_times = []
        for train_index, test_index in dataset.sss.split(dataset.X, dataset.y):
            # Split into train / test data using the SSS as a guide
            X_train, X_val = dataset.X.iloc[train_index], dataset.X.iloc[test_index]
            y_train, y_val = dataset.y.iloc[train_index], dataset.y.iloc[test_index]

            # Fit the classifier to the training data
            model.fit(X_train, y_train)

            # Predict on the validation data and calculate run-time
            start_time = time.time()
            y_pred = model.predict(X_val)
            end_time = time.time()
            run_times.append(end_time - start_time)

            # Calculate the accuracy and append to the list
            accuracy = accuracy_score(y_val, y_pred)
            accuracies.append(accuracy)
            thr_acc.append(accuracy_score(y_val[y_val == 2.0], y_pred[y_val == 2.0]) * 100)
            if dataset.num_of_labels == 3:
                tmr_acc.append(accuracy_score(y_val[y_val == 1.0], y_pred[y_val == 1.0]) * 100)
            tlr_acc.append(accuracy_score(y_val[y_val == 0.0], y_pred[y_val == 0.0]) * 100)

            # Calculate the confusion matrix and append to the list
            cm = confusion_matrix(y_val, y_pred, normalize='true')
            confusion_matrices.append(cm)

        # Calculate the mean and standard deviation of the accuracy across all splits
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(
            f"Mean Accuracy for {dataset.num_of_labels} level classification: {mean_accuracy:.3f} +/- {std_accuracy:.3f}")

        # print results
        print(f"    Accuracy for True High Rate (THR): {np.mean(thr_acc):.2f}%")
        if dataset.num_of_labels == 3:
            print(f"    Accuracy for True Medium Rate (TMR): {np.mean(tmr_acc):.2f}%")
        print(f"    Accuracy for True Low Rate (TLR): {np.mean(tlr_acc):.2f}%")
        print(f"    Average run-time: {np.mean(run_times)}")

        # Calculate the average confusion matrix
        mean_cm = np.mean(confusion_matrices, axis=0)

        # Calculate precision, recall, and F1 score
        precision = np.diag(mean_cm) / np.sum(mean_cm, axis=0)
        precision_mean=np.mean(precision)
        recall = np.diag(mean_cm) / np.sum(mean_cm, axis=1)
        recall_mean=np.mean(recall)
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score_mean=np.mean(f1_score)
        # Print the average F1 score
        print(f"    Average F1 Score: {np.mean(f1_score)}")
        print(f"    Average recall: {np.mean(recall)}")
        print(f"    Average precision: {np.mean(precision)}")

        # plot confusion matrix if desired
        if plot_CM:
            ML_Utilities.plot_confustion_matrix(dataset.num_of_labels, cm=mean_cm,model_type=model)
        if dataset.num_of_labels == 3:
            results = pd.DataFrame({
                'accuracy': [mean_accuracy * 100],
                'std_accuracy': [std_accuracy * 100],
                'precision': [precision_mean * 100],
                'recall': [recall_mean * 100],
                'f1_score_mean': [f1_score_mean * 100],
                'True High Rate(THR)':np.mean(thr_acc),
                'True Medium Rate(THR)': np.mean(tmr_acc),
                'True Low Rate(THR)': np.mean(tlr_acc)})
        if dataset.num_of_labels == 2:
            results = pd.DataFrame({
                'accuracy': [mean_accuracy * 100],
                'std_accuracy': [std_accuracy * 100],
                'precision': [precision_mean * 100],
                'recall': [recall_mean * 100],
                'f1_score_mean': [f1_score_mean * 100],
                'True High Rate(THR)':np.mean(thr_acc),
                'True Low Rate(THR)': np.mean(tlr_acc)})
        results.to_csv(self.data_path+f'\{self.iteration}_k_{dataset.num_of_labels}_labels_table score_SVM.csv')

        # return as percentage
        return results
    # Naive Bayes Classifier
    def NB_tuner(self):
        print("\nNaive Bayes Classifier:")

        # calculate training class probabilities for Gaussian NB
        low = self.dataset_three_level.y_train.value_counts()[1.0] / len(self.dataset_three_level.y_train)
        if self.dataset_three_level.num_of_labels == 3:
            medium = self.dataset_three_level.y_train.value_counts()[2.0] / len(self.dataset_three_level.y_train)
        high = self.dataset_three_level.y_train.value_counts()[3.0] / len(self.dataset_three_level.y_train)
        Gaussian_priors = [low, medium, high] if self.dataset_three_level.num_of_labels == 3 else [low, high]

        # create a dictionary of classifiers
        classifiers = {'Multinomial': MultinomialNB(),
                       'Bernoulli': BernoulliNB(),
                       'Complement': ComplementNB(),
                       'Gaussian': GaussianNB()}

        # set up a parameter grid for each classifier
        param_grids = {'Multinomial': {'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
                       'Bernoulli': {'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                                     'binarize': [0.0, 0.5, 1.0]},
                       'Complement': {'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
                       'Gaussian': {'priors': [Gaussian_priors]}
                       }

        accuracies_binary = []
        std_devs_binary = []
        accuracies_three = []
        std_devs_three = []
        for name, nb in classifiers.items():
            # perform a grid search to find the best hyperparameters for each classifier
            grid_search = GridSearchCV(nb, param_grid=param_grids[name])
            grid_search.fit(self.dataset_three_level.X_train, self.dataset_three_level.y_train)

            # print the best hyperparameters for each classifier
            print('\nBest hyperparameters for {}: {}'.format(name, grid_search.best_params_))

            # perform a 5 fold cross validation using the best grid search for both 2 and 3 level classification
            accuracy, std_dev = self.evaluate(grid_search.best_estimator_, dataset=self.dataset_three_level,
                                              plot_CM=True)
            accuracies_three.append(accuracy)
            std_devs_three.append(std_dev)
            if name == 'Gaussian':
                grid_search.best_estimator_.priors = [0.5, 0.5]
            accuracy, std_dev = self.evaluate(grid_search.best_estimator_, dataset=self.dataset_binary, plot_CM=True)
            accuracies_binary.append(accuracy)
            std_devs_binary.append(std_dev)

        # Set the width of each bar
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        r1 = np.arange(len(classifiers.keys()))
        r2 = [x + bar_width for x in r1]

        # Create the bar chart
        plt.bar(r1, accuracies_binary, width=bar_width, color='paleturquoise', yerr=std_devs_binary, capsize=5,
                label='Binary classification')
        plt.bar(r2, accuracies_three, width=bar_width, color='darkslategray', yerr=std_devs_three, capsize=5,
                label='Three-level classification')

        # Add labels, title, and legend
        plt.xlabel('Classifier')
        plt.xticks([r + bar_width / 2 for r in range(len(classifiers.keys()))], classifiers.keys())
        plt.ylabel('Accuracy (%)')
        plt.ylim([0, 100])
        plt.legend()

        # Show the bar chart
        plt.show()

    # Post tuned Naive Bayes Classifer
    def NB_classifier(self):
        # binary classification
        low = self.dataset_binary.y_train.value_counts()[1.0] / len(self.dataset_binary.y_train)
        high = self.dataset_binary.y_train.value_counts()[3.0] / len(self.dataset_binary.y_train)
        Gaussian_priors = [low, high]
        nb = GaussianNB(priors=Gaussian_priors)
        self.evaluate(nb, self.dataset_binary)

        # three-level classification
        low = self.dataset_three_level.y_train.value_counts()[1.0] / len(self.dataset_three_level.y_train)
        medium = self.dataset_three_level.y_train.value_counts()[2.0] / len(self.dataset_three_level.y_train)
        high = self.dataset_three_level.y_train.value_counts()[3.0] / len(self.dataset_three_level.y_train)
        Gaussian_priors = [low, medium, high]
        nb = GaussianNB(priors=Gaussian_priors)
        self.evaluate(nb, self.dataset_three_level)

    # Support Vector Machine Classifier
    def SVM_tuner(self):
        print("\nSVM Classifier:")

        # create a dictionary of kernels
        kernels = {'Linear': 'linear',
                   'Polynomial': 'poly',
                   'Radial basis function': 'rbf',
                   'Sigmoid': 'sigmoid'}

        # set up a parameter grid for each kernel
        param_grids = {'Linear': {'C': [0.1, 1, 10, 100, 1000]},
                       'Polynomial': {'C': [0.1, 1, 10, 100, 1000],
                                      'degree': [2, 3, 4],
                                      'gamma': [0.1, 0.01]},
                       'Radial basis function': {'C': [0.1, 1, 10, 100, 1000],
                                                 'gamma': [0.1, 0.01, 0.001]},
                       'Sigmoid': {'C': [0.1, 1, 10, 100, 1000],
                                   'gamma': [0.1, 0.01, 0.001],
                                   'coef0': [0.1, 0.01, 0.001]}
                       }

        accuracies_binary = []
        std_devs_binary = []
        accuracies_three = []
        std_devs_three = []
        dataset = self.dataset_binary
        for name, kernel in kernels.items():
            # perform a grid search to find the best hyperparameters for each kernel
            grid_search = GridSearchCV(SVC(kernel=kernel), param_grid=param_grids[name], n_jobs=self.n_jobs)
            grid_search.fit(dataset.X_train, dataset.y_train)

            # print the best hyperparameters for each kernel
            print('\nBest hyperparameters for {}: {}'.format(name, grid_search.best_params_))

            # perform a 5 fold cross validation using the best grid search for both 2 and 3 level classification
            accuracy, std_dev = self.evaluate(grid_search.best_estimator_, dataset=self.dataset_three_level,
                                              plot_CM=True)
            accuracies_three.append(accuracy)
            std_devs_three.append(std_dev)
            accuracy, std_dev = self.evaluate(grid_search.best_estimator_, dataset=self.dataset_binary, plot_CM=True)
            accuracies_binary.append(accuracy)
            std_devs_binary.append(std_dev)

        # Set the width of each bar
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        r1 = np.arange(len(kernels.keys()))
        r2 = [x + bar_width for x in r1]

        # Create the bar chart
        plt.bar(r1, accuracies_binary, width=bar_width, color='lightsalmon', yerr=std_devs_binary, capsize=5,
                label='Binary classification')
        plt.bar(r2, accuracies_three, width=bar_width, color='orangered', yerr=std_devs_three, capsize=5,
                label='Three-level classification')

        # Add labels, title, and legend
        plt.xlabel('Classifier')
        plt.xticks([r + bar_width / 2 for r in range(len(kernels.keys()))], kernels.keys())
        plt.ylabel('Accuracy (%)')
        plt.ylim([0, 100])
        plt.legend()

        # Show the bar chart
        plt.show()

    # Post-tuned Support Vector Machine Classifier
    def SVM_classifier(self):
        # binary classification

        # svm = SVC(kernel='rbf', C=1000, gamma=0.1, random_state=15)
        svm = SVC(kernel='poly', degree=2)
        results_binary=self.evaluate(svm, self.dataset_binary)


        # three-level classification
        # svm = SVC(kernel='poly', C=1000, degree=4, gamma=0.1, random_state=15)
        svm = SVC(kernel='poly', degree=2)
        three_level=self.evaluate(svm, self.dataset_three_level)

        return results_binary,three_level
        # Random Forest Tuner

    def RF_tuner(self):
        print("\nRandom Forest Classifier:")

        # define the criterion options
        criterions = {'Gini Index': 'gini',
                      'Entropy': 'entropy',
                      'Logarithmic Loss': 'log_loss'}

        # define the parameter grid
        param_grid = {'n_estimators': [10, 50, 100, 250, 500],
                      'max_depth': [None, 2, 3, 4, 5, 7],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'max_features': [1, 'sqrt', 'log2', None],
                      'bootstrap': [True, False]}

        accuracies_binary = []
        std_devs_binary = []
        accuracies_three = []
        std_devs_three = []
        dataset = self.dataset_three_level

        for name, criterion in criterions.items():
            # create a random forest classifier with the current criterion
            rf = RandomForestClassifier(criterion=criterion)

            # perform a grid search to find the best hyperparameters for the classifier
            grid_search = GridSearchCV(rf, param_grid=param_grid, n_jobs=self.n_jobs)
            grid_search.fit(dataset.X_train, dataset.y_train)

            # print the best hyperparameters for the classifier
            print('\nBest hyperparameters for {}: {}'.format(name, grid_search.best_params_))

            # evaluate the classifier on the test set
            accuracy, std_dev = self.evaluate(grid_search.best_estimator_, dataset=self.dataset_three_level,
                                              plot_CM=True)
            accuracies_three.append(accuracy)
            std_devs_three.append(std_dev)
            accuracy, std_dev = self.evaluate(grid_search.best_estimator_, dataset=self.dataset_binary, plot_CM=True)
            accuracies_binary.append(accuracy)
            std_devs_binary.append(std_dev)

        # Set the width of each bar
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        r1 = np.arange(len(criterions.keys()))
        r2 = [x + bar_width for x in r1]

        # Create the bar chart
        plt.bar(r1, accuracies_binary, width=bar_width, color='greenyellow', yerr=std_devs_binary, capsize=5,
                label='Binary classification')
        plt.bar(r2, accuracies_three, width=bar_width, color='forestgreen', yerr=std_devs_three, capsize=5,
                label='Three-level classification')

        # Add labels, title, and legend
        plt.xlabel('Criterion')
        plt.xticks([r + bar_width / 2 for r in range(len(criterions.keys()))], criterions.keys())
        plt.ylabel('Accuracy (%)')
        plt.ylim([0, 100])
        plt.legend()

        # Show the bar chart
        plt.show()

    # Post-tuned Random Forest Classifier
    def RF_classifier(self):
        params = {'criterion': 'gini', 'bootstrap': False, 'max_depth': None, 'max_features': 'log2',
                  'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
        rf = RandomForestClassifier(**params, random_state=15)

        # binary classification
        self.evaluate(rf, self.dataset_binary)

        # three-level classification
        self.evaluate(rf, self.dataset_three_level)