from util import *

# Add your import statements here




class InformationRetrieval_TFIDF():

	def __init__(self):
		self.index = None
		self.vectorizer = None
		self.docIDs = None

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
		total = []
		for i in range(len(docIDs)):
			curr = docs[i]
			s = listToString(curr)
			total.append(s)
		# print(total[0])
		# print('Finished array making')
		vectorizer = TfidfVectorizer()
		X = vectorizer.fit_transform(total)
		self.index = np.array(X.todense())
		self.vectorizer = vectorizer
		self.docIDs = docIDs
		# print('Vectorizing done')


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

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
		e = 0
		# print(len(queries))
		for query in queries:
			e = e + 1
			# print(e)
			sim = {}
			curr = []
			s = listToString(query)
			q = [s]
			test_array = self.vectorizer.transform(q)
			X = self.index
			a = np.array(test_array.todense())
			a = a.reshape(-1)
			for i in range(len(self.docIDs)):
				b = X[i]
				#print(a.shape)
				#print(b.shape)
				cosine = cx(a, b)
				sim[self.docIDs[i]] = cosine
			sorted_docs = dict(sorted(sim.items(),key=lambda item: item[1],reverse=True))
			final_list = list(sorted_docs.keys())
			doc_IDs_ordered.append(final_list)
		# print(doc_IDs_ordered[0])
		return doc_IDs_ordered



