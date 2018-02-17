import abc
from abc import abstractmethod

class Retrieval:
	__metaclass__ = abc.ABCMeta
	@classmethod
	def __init__(self):
		pass

	@classmethod
	def getLongSnippets(self, question):
		return self.getSnippets(question, False)

	@classmethod
	def getShortSnippets(self, question):
		return self.getSnippets(question, True)

	@classmethod
	def getSnippets(self, question, isShortSnippet):
		snippetSelection = 'short_snippets' if isShortSnippet else 'long_snippets'
		snippets = question['contexts'][snippetSelection]
		return ' '.join(snippets)