from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score


X, y = datasets.load_iris(return_X_y=True)
n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
clf = svm.SVC(kernel='linear', C=1, random_state=42)
cross_val_score(clf, X, y, cv=cv)
