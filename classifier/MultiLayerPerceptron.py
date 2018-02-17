from Classifier import Classifier
from sklearn.neural_network import MLPClassifier

class MultiLayerPerceptron(Classifier):

    def buildClassifier(self, X_features, Y_train):
        clf = MLPClassifier()
        return clf.fit(X_features, Y_train)