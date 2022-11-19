from util import *
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
# Add your import statements here




class InformationRetrieval_LSA():

	def __init__(self):
		self.docIDs = []
		self.docs = []
		self.vocab = None
		self.vectorizer = None
		self.lsa = None



	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the latent topics in the corpus and stores the tranformer tool in 'lsa'

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
		self.k = 575
		vectorizer = TfidfVectorizer()
		self.docs = []
		for doc in docs:
			self.docs.append(listToString(doc))
		X = vectorizer.fit_transform(self.docs)
		#print("Count vectorizer done")
		self.vectorizer = vectorizer
		lsa = TruncatedSVD(n_components = self.k,random_state=42,n_iter=10)
		X1 = lsa.fit_transform(X)
		self.docs = X1
		self.lsa = lsa
		#print("LSA fit done")
		self.vocab = vectorizer.get_feature_names()

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query,using latent topics found using LSA

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
		for query in queries:
			sim = {}
			query = [listToString(query)]
			#print(query)
			arr = self.vectorizer.transform(query)
			#print(arr)
			arr1 = self.lsa.transform(arr)
			for i in range(len(self.docIDs)):
				a = arr1.reshape(-1)
				b = np.array((self.docs[i]))
				#print(a.shape)
				#print(b.shape)
				cosine = cx(a, b)
				sim[self.docIDs[i]] = cosine
			sorted_docs = dict(sorted(sim.items(),key=lambda item: item[1],reverse=True))
			final_list = list(sorted_docs.keys())
			doc_IDs_ordered.append(final_list)
		return doc_IDs_ordered




