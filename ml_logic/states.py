import pandas as pd
import numpy as np


class Token():

    def __init__(self,
                 data_fp='../Data/watson_healthcare_modified.csv',
                 upsampling_method='SMOTE',
                 cols = None,
                 ):


        self.data = pd.read_csv(data_fp)
        self.upsampling_method = upsampling_method
        self.cols = [] if None else cols
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_train = None

        self.accuracy = -1
        self.recall = -1
        self.precision = -1
        self.f1 = -1

        def preprocess(self):
            return -1

        def upsample(self):
            return -1

        def logistic_regression(self):
            return -1

        def svm(self):
            return -1

        def knn(self):
            return -1

        def xgboost(self):
            return -1

        def plot(self):
            roc_curve = None
            return -1
