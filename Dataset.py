class Dataset():
    def __init__(self, X, y, X_train, y_train, X_test, y_test, sss, num_of_labels):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.sss = sss
        self.num_of_labels = num_of_labels