from util import *
from collections import defaultdict
import numpy as np
# Add your import statements here




class InformationRetrieval_LDA():

	def __init__(self):
		self.docIDs = None
		self.word_dict = None
		self.doc_dict = None
		self.color_dict = None
		self.topic_imp = None
		self.alpha = None
		self.beta = None
		self.k = None

	def assign_random(self,doc,doc_id,k):
		"""
		Initializes the distributions using pseudo random values

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		arg3 : int
			Number of latent topics
		Returns
		-------
		None
		"""
		m = []
		for word in doc:
			#print(word + "Hello")
			#print(doc_id)
			d1 = self.word_dict[word]
			d2 = self.doc_dict[doc_id]
			#print(d1)
			#print(d2)
			dist1 = [(d1[i]*d2[i])for i in range(k)]
			dist2 = [dist1[i]/self.topic_imp[i] for i in range(k)]
			dist = np.array(dist2)
			col = np.random.multinomial(1, dist / dist.sum()).argmax()
			self.doc_dict[doc_id][col] += 1
			self.word_dict[word][col] += 1
			self.topic_imp[col] += 1
			m.append(col)
		self.color_dict[doc_id] = m
		return None


	def gibbs_iteration(self,docs,doc_ids,k):
		"""
		Improves the document-topic and topic-word distributions using one iteration of Gibbs sampling

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		arg3 : int
			Number of latent topics
		Returns
		-------
		None
		"""
		for i  in range(len(docs)):
			doc = docs[i]
			doc_id = doc_ids[i]
			asgn = self.color_dict[doc_id]
			for j in range(len(doc)):
				word = doc[j]
				col = asgn[j]
				wd = self.word_dict[word]
				dd = self.doc_dict[doc_id]
				wd[col] = wd[col]-1
				dd[col] = dd[col]-1
				self.topic_imp[col] -= 1
				dist1 = [(wd[t] * dd[t])/self.topic_imp[t] for t in range(k)]
				dist = np.array(dist1)
				newcol = np.random.multinomial(1, dist / dist.sum()).argmax()
				asgn[j] = newcol
				wd[newcol] += 1
				dd[newcol] +=1
				self.topic_imp[newcol] += 1
		return None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

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
		self.word_dict = defaultdict(dict)
		self.doc_dict = defaultdict(dict)
		self.topic_imp = {}
		self.color_dict = {}
		doc_ids = docIDs
		self.topic_imp = {}
		self.alpha = 0.1
		self.beta = 0.1
		self.k = 50
		self.iter = 50
		M = len(vocab)
		#print(M)
		#with open('vocab.txt','w') as f:
		#	f.write(str(vocab))
		for word in vocab:
			for c in range(self.k):
				self.word_dict[word][c] = self.beta
		for doc_id in docIDs:
			for c in range(self.k):
				self.doc_dict[doc_id][c] = self.alpha
		for i in range(self.k):
			self.topic_imp[i] = M*self.beta
		#print("Starting Random Assignment")
		for i in range(len(doc_ids)):
			self.assign_random(docs[i],doc_ids[i],self.k)
		#print("Random Assignment Done")
		for i in range(self.iter):
			#print("Iteration " + str(i))
			self.gibbs_iteration(docs,doc_ids,self.k)
		self.docIDs = doc_ids


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query, using latent topics found by LDA

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
		# print('Entered Ranking')
		doc_IDs_ordered = []
		#for i in range(10):
		#	print(self.doc_dict[self.docIDs[i]])
		e = 1
		for query in queries:
			#print("Looking at query "+ str(e))
			e = e+1
			sim = {}
			curr = {}
			for j in range(self.k):
				curr[j] = 0
			for word in query:
				if word in vocab:
					d1 = self.word_dict[word]
					for j in range(self.k):
						curr[j] = curr[j] + d1[j]/self.topic_imp[j]
			for i in range(len(self.docIDs)):
				a = [curr[u] for u in range(self.k)]
				b = list(self.doc_dict[self.docIDs[i]].values())
				a = np.array(a)
				b = np.array(b)
				cosine = cx(a, b)
				sim[self.docIDs[i]] = cosine
			sorted_docs = dict(sorted(sim.items(),key=lambda item: item[1],reverse=True))
			final_list = list(sorted_docs.keys())
			doc_IDs_ordered.append(final_list)
		return doc_IDs_ordered




