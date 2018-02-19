from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB


class Tf_Idf_Featurizer(Featurizer):

	def getFeatureRepresentation(self, X_train, X_val):
		vectorizer = TfidfVectorizer()
		X_train_weights = vectorizer.fit_transform(X_train)
		X_val_weights = vectorizer.transform(X_val)
		return X_train_weights, X_val_weights

	def getName(self):
		return "Tf-Idf Featurizer"