import sys
import json
from sklearn.externals import joblib

from Retrieval import Retrieval
from featurizer import Featurizer
from featurizer import CountFeaturizer
from classifier import Classifier
from classifier import MultinomialNaiveBayes
from Evaluator import Evaluator


class Pipeline(object):
    def __init__(self, trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance):
        self.retrievalInstance = retrievalInstance
        self.featurizerInstance = featurizerInstance
        self.classifierInstance = classifierInstance

        trainfile = open(trainFilePath, 'r')
        self.trainData = json.load(trainfile)
        trainfile.close()

        valfile = open(valFilePath, 'r')
        self.valData = json.load(valfile)
        valfile.close()

        self.question_answering()

    def makeXY(self, dataQuestions):
        X = []
        Y = []
        for question in dataQuestions:

            long_snippets = self.retrievalInstance.getLongSnippets(question)
            short_snippets = self.retrievalInstance.getShortSnippets(question)

            X.append(short_snippets)
            Y.append(question['answers'][0])

        return X, Y

    def question_answering(self):
        # dataset_type = self.trainData['origin']
        # candidate_answers = self.trainData['candidates']
        X_train, Y_train = self.makeXY(self.trainData['questions'][0:10])
        X_val, Y_val_true = self.makeXY(self.valData['questions'])

        # Featurization
        X_features_train, X_features_val = self.featurizerInstance.getFeatureRepresentation(
            X_train, X_val)
        self.clf = self.classifierInstance.buildClassifier(
            X_features_train, Y_train)

        # Prediction
        Y_val_pred = self.clf.predict(X_features_val)

        self.evaluatorInstance = Evaluator()
        a = self.evaluatorInstance.getAccuracy(Y_val_true, Y_val_pred)
        p, r, f = self.evaluatorInstance.getPRF(Y_val_true, Y_val_pred)
        print("Accuracy: "+str(a))
        print("Precision: " + str(p))
        print("Recall: " + str(r))
        print("F-measure: " + str(f))


if __name__ == '__main__':
    trainFilePath = sys.argv[1] # C:\wl-shared\deiis\quasar-s_dev_formatted.json
    valFilePath = sys.argv[2]
    retrievalInstance = Retrieval()
    featurizerInstance = CountFeaturizer.CountFeaturizer()
    classifierInstance = MultinomialNaiveBayes.MultinomialNaiveBayes()
    trainInstance = Pipeline(trainFilePath, valFilePath,
                             retrievalInstance, featurizerInstance, classifierInstance)
