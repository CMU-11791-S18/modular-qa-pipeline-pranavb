from Classifier import Classifier
from sklearn import svm

class SupportVectorMachine(Classifier):

    def buildClassifier(self, X_features, Y_train):
        clf = svm.SVC()
        return clf.fit(X_features, Y_train)