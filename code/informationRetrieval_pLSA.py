from util import *
from collections import defaultdict
from pylab import random
from numpy import zeros, int8, log
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from numpy import savez_compressed,load
# Add your import statements here




class InformationRetrieval_pLSA():

	def __init__(self):
		self.docIDs = []
		self.docs = []
		self.vocab = None
		self.vectorizer = None
		self.lsa = None
		self.k = None
		self.maxIteration = 30
		self.threshold = 10.0
		self.topicWordsNum = 10
		self.lamda = None
		self.theta = None
		self.p = None
		self.X = None
		



	def initializeParameters(self):
		"""
		Initialize all distributions, theta and lamda
		----------
		None
		-------
		None
		"""
		for i in range(0, self.X.shape[0]):
			normalization = sum(self.lamda[i, :])
			for j in range(0, self.k):
				self.lamda[i, j] /= normalization;
		
		for i in range(0, self.k):
			normalization = sum(self.theta[i, :])
			for j in range(0, self.X.shape[1]):
				self.theta[i, j] /= normalization;

	def EStep(self):
		"""
		Expectation Step

		Build the overall distribution (p) from lamda (document topic) and theta (topic word) distribution
		----------
		None
		-------
		None
		"""
		for i in range(0, self.X.shape[0]):
			for j in range(0, self.X.shape[1]):
				denominator = 0;
				for k in range(0, self.k):
					self.p[i, j, k] = self.theta[k, j] * self.lamda[i, k];
					denominator += self.p[i, j, k];
				if denominator == 0:
					for k in range(0, self.k):
						self.p[i, j, k] = 0;
				else:
					for k in range(0, self.k):
						self.p[i, j, k] /= denominator;

	def MStep(self):
		"""
		Build the topic word distribution and the document topic distribtion from
		document word distribution and the total distribution (p)

		Parameters
		----------
		None
		Returns
		-------
		None
		"""
		for k in range(0, self.k):
			denominator = 0
			for j in range(0, self.X.shape[1]):
				self.theta[k, j] = 0
				for i in range(0, self.X.shape[0]):
					self.theta[k, j] += self.X[i, j] * self.p[i, j, k]
				denominator += self.theta[k, j]
			if denominator == 0:
				for j in range(0, self.X.shape[1]):
					self.theta[k, j] = 1.0 / self.X.shape[1]
			else:
				for j in range(0, self.X.shape[1]):
					self.theta[k, j] /= denominator
		for i in range(0, self.X.shape[0]):
			for k in range(0, self.k):
				self.lamda[i, k] = 0
				denominator = 0
				for j in range(0, self.X.shape[1]):
					self.lamda[i, k] += self.X[i, j] * self.p[i, j, k]
					denominator += self.X[i, j];
				if denominator == 0:
					self.lamda[i, k] = 1.0 / self.k
				else:
					self.lamda[i, k] /= denominator

			

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index by extracting latent topics from the document's words using pLSA

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		self.docIDs = docIDs
		self.k = 25
		self.maxIteration = 30
		self.threshold = 10.0
		self.topicWordsNum = 10
		vectorizer = TfidfVectorizer(max_features=3000)
		self.docs = []
		for doc in docs:
			self.docs.append(listToString(doc))
		X = vectorizer.fit_transform(self.docs)
		self.X = np.array(X.todense())
		self.lamda = random([self.X.shape[0], self.k])
		self.theta = random([self.k, self.X.shape[1]])
		self.vectorizer = vectorizer
		self.p = zeros([self.X.shape[0], self.X.shape[1], self.k])
		self.initializeParameters()
		for i in range(0, self.maxIteration):
			self.EStep()
			self.MStep()

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query,using latent topics found by pLSA

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""
		doc_IDs_ordered = []
		for query in queries:
			sim = {}
			query = [listToString(query)]
			q = self.vectorizer.transform(query)
			arr = np.array(q.todense())
			a1 = zeros((self.k))
			arr = arr.reshape(-1)
			for i in range(arr.shape[0]):
				a1 = a1 + arr[i]*self.theta[:,i]
			for i in range(len(self.docIDs)):
				a = a1
				b = self.lamda[i]
				cosine = cx(a, b)
				sim[self.docIDs[i]] = cosine
			sorted_docs = dict(sorted(sim.items(),key=lambda item: item[1],reverse=True))
			final_list = list(sorted_docs.keys())
			doc_IDs_ordered.append(final_list)
		return doc_IDs_ordered




