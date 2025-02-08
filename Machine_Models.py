import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('processed_data_agg.csv')
X = data.drop(columns=['Stress Level'])
y = data['Stress Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = SMOTE(sampling_strategy='not majority').fit_resample(X_train, y_train)



sfs = SequentialFeatureSelector(rf_feature_selector, n_features_to_select="auto", direction="forward", cv=5)
sfs.fit(X_train, y_train)
X_train_selected = sfs.transform(X_train)
X_test_selected = sfs.transform(X_test)

# RF
rf = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3)
rf_grid_search.fit(X_train_selected, y_train)
best_rf = rf_grid_search.best_estimator_

# Cross-validation accuracy for Random Forest
rf_cv_scores = cross_val_score(best_rf, X_train_selected, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))
rf_accuracy = rf_cv_scores.mean()

# Confusion matrix for Random Forest
rf_y_pred = best_rf.predict(X_test_selected)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)

# NN
nn = MLPClassifier()
nn_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}
nn_grid_search = GridSearchCV(estimator=nn, param_grid=nn_param_grid, cv=3)
nn_grid_search.fit(X_train_selected, y_train)
best_nn = nn_grid_search.best_estimator_


nn_cv_scores = cross_val_score(best_nn, X_train_selected, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))
nn_accuracy = nn_cv_scores.mean()

nn_y_pred = best_nn.predict(X_test_selected)
nn_conf_matrix = confusion_matrix(y_test, nn_y_pred)

# XGBoost 
xgb = XGBClassifier()
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}
xgb_grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=3)
xgb_grid_search.fit(X_train_selected, y_train)
best_xgb = xgb_grid_search.best_estimator_


xgb_cv_scores = cross_val_score(best_xgb, X_train_selected, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))
xgb_accuracy = xgb_cv_scores.mean()


xgb_y_pred = best_xgb.predict(X_test_selected)
xgb_conf_matrix = confusion_matrix(y_test, xgb_y_pred)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_selected)
train_clusters = kmeans.predict(X_train_selected)
cluster_accuracy = accuracy_score(y_train, train_clusters)


cluster_conf_matrix = confusion_matrix(y_train, train_clusters)


print("Random Forest CV-10 Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:\n", rf_conf_matrix)
print("Neural Network CV-10 Accuracy:", nn_accuracy)
print("Neural Network Confusion Matrix:\n", nn_conf_matrix)
print("XGBoost CV-10 Accuracy:", xgb_accuracy)
print("XGBoost Confusion Matrix:\n", xgb_conf_matrix)
print("Clustering Accuracy:", cluster_accuracy)
print("Clustering Confusion Matrix:\n", cluster_conf_matrix)
