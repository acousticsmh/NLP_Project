from util import *
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
# Add your import statements here




class InformationRetrieval_BM25():

	def __init__(self):
		self.docIDs = []
		self.docs = []
		self.vocab = None
		self.vectorizer = None
		self.avg = 0.0
		self.lengths = []
		self.X = None



	def buildIndex(self, docs, docIDs):
		"""
		Builds the Document-word Count matrix and stores it in self.X. Stores the count vectorizer in self.vectorizer

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
		vectorizer = CountVectorizer()
		self.docs = []
		tot = 0.0
		for doc in docs:
			self.lengths.append(len(doc))
			tot = tot + len(doc)
			self.docs.append(listToString(doc))
		avg = tot/len(self.docIDs)
		X = vectorizer.fit_transform(self.docs)
		self.vectorizer = vectorizer
		self.vocab = vectorizer.vocabulary_
		self.X = np.array(X.todense())
		self.avg = avg
		#print("Vectorizer done")
		#print("Avg document length is " + str(self.avg))

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query, using the BM25 measure

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
		#print('Entered Ranking')
		doc_IDs_ordered = []
		k5 = 2.0
		b = 0.75
		e = 0
		for query in queries:
			#print("Looking at query " + str(e))
			sim = {}
			idf = np.zeros((len(query)))
			j = 0
			for word in query:
				if word in self.vocab:
					num = self.vocab[word]
					arr = self.X[:,num]
					n = np.count_nonzero(arr)
					s = IDF(len(self.docs),n)
					idf[j] = s
				else:
					idf[j] = 0.0
				j = j + 1
			#print(idf)
			for i in range(len(self.docIDs)):
				tf = np.zeros((len(query)))
				k = 0
				for word in query:
					if word in self.vocab:
						num = self.vocab[word]
						f = self.X[i][num]
						sc = fun(f,k5,b,self.lengths[i],self.avg)
						tf[k] = sc
					else:
						tf[k] = 0.0
					k = k + 1
				score = np.dot(idf,tf)*1.0
				sim[self.docIDs[i]] = score
			sorted_docs = dict(sorted(sim.items(),key=lambda item: item[1],reverse=True))
			final_list = list(sorted_docs.keys())
			doc_IDs_ordered.append(final_list)
		return doc_IDs_ordered




