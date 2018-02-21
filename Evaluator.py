import abc
from abc import abstractmethod
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

class Evaluator:
	__metaclass__ = abc.ABCMeta
	@classmethod
	def __init__(self): #constructor for the abstract class
		pass

	#This is a class method that gets accuracy of the model
	@classmethod
	def getAccuracy(self, Y_true, Y_pred):
		accuracy = accuracy_score(Y_true, Y_pred)

		self.assessPredictions(Y_true, Y_pred)
		return accuracy
	
	# Code to determine matches for the relative performance analysis
	@classmethod
	def assessPredictions(self, Y_true, Y_pred):
		print('\nNum of results: {}'.format(len(Y_true)))

		i = 0
		numOfMatches = 0
		matches = []
		misses = []

		print('\nValid : Predicted')
		for validToken in Y_true:
			if validToken == Y_pred[i]:
				numOfMatches += 1
				matches.append(validToken)
				print('{} : {}'.format(validToken, Y_pred[i]))
			else:
				misses.append(validToken)
				print('\t\t{} : {}'.format(validToken, Y_pred[i]))
			i += 1
		
		print('\nNumber of matches: {}'.format(numOfMatches))

		print('\nMatches')
		print(matches)

		print('\nMisses')
		print(misses)
	
	#This is a class method that gets precision, recall and f-measure of the model	
	@classmethod
	def getPRF(self, Y_true, Y_pred):
		prf = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
		precision = prf[0]
		recall = prf[1]
		f_measure = prf[2]
		return precision, recall, f_measure
