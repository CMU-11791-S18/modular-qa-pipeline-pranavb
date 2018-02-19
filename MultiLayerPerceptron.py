from Classifier import Classifier
from sklearn.neural_network import MLPClassifier

class MultiLayerPerceptron(Classifier):

    def buildClassifier(self, X_features, Y_train):
        return MLPClassifier().fit(X_features, Y_train)

    def getName(self):
        return 'Multi Layer Perceptron Classifier'