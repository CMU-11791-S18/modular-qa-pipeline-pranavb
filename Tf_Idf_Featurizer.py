from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer

# @author pbolar
class Tf_Idf_Featurizer(Featurizer):
    
    def getFeatureRepresentation(self, X_train, X_val):
        transformer = TfidfVectorizer()
        x_train_weights = transformer.fit_transform(X_train)
        x_val_weights = transformer.fit(X_val)
        return (x_train_weights, x_val_weights)