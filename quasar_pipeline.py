import sys
import json
from sklearn.externals import joblib

from Retrieval import Retrieval
from Evaluator import Evaluator

from Featurizer import Featurizer
from CountFeaturizer import CountFeaturizer
# from Tf_Idf_Featurizer import Tf_Idf_Featurizer

from Classifier import Classifier
# from MultinomialNaiveBayes import MultinomialNaiveBayes
from SupportVectorMachine import SupportVectorMachine
# from MultiLayerPerceptron import MultiLayerPerceptron


class Pipeline(object):
    def __init__(self, trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance):
        self.retrievalInstance = retrievalInstance
        self.featurizerInstance = featurizerInstance
        self.classifierInstance = classifierInstance

        print('Parsing the training file')
        trainfile = open(trainFilePath, 'r')
        self.trainData = json.load(trainfile)
        trainfile.close()
        print('Completed parsing')

        print('Parsing the validation file')
        valfile = open(valFilePath, 'r')
        self.valData = json.load(valfile)
        valfile.close()
        print('Completed parsing')

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
        print('Featurizing')
        X_features_train, X_features_val = self.featurizerInstance.getFeatureRepresentation(
            X_train, X_val)
        print('Classifying')
        self.clf = self.classifierInstance.buildClassifier(
            X_features_train, Y_train)

        # Prediction
        print('Running the classifier\'s prediction algo')
        Y_val_pred = self.clf.predict(X_features_val)

        # Evaluation
        print('Evaluating')
        self.evaluatorInstance = Evaluator()
        a = self.evaluatorInstance.getAccuracy(Y_val_true, Y_val_pred)
        p, r, f = self.evaluatorInstance.getPRF(Y_val_true, Y_val_pred)

        print('\n------------- RESULTS -------------')
        print("Accuracy:\t"  + str(a))
        print("Precision:\t" + str(p))
        print("Recall:\t\t"  + str(r))
        print("F-measure:\t" + str(f))


if __name__ == '__main__':
    trainFilePath = sys.argv[1] # C:\wl-shared\deiis\quasar-s_train_formatted.json
    valFilePath = sys.argv[2] # C:\wl-shared\deiis\quasar-s_dev_formatted.json
    retrievalInstance = Retrieval()
    featurizerInstance = CountFeaturizer()
    classifierInstance = SupportVectorMachine()
    trainInstance = Pipeline(trainFilePath, valFilePath,
                             retrievalInstance, featurizerInstance, classifierInstance)
