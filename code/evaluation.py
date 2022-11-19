from util import *

# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""
		s = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				s = s + 1
		precision = float(s/k)
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""
		meanPrecision = 0.0

		for i in range(len(query_ids)):
			true_docs = []
			curr_doc_IDs_ordered = doc_IDs_ordered[i]
			for items in qrels:
				if query_ids[i] == int(items['query_num']):
					true_docs.append(int(items['id']))
			cp = self.queryPrecision(curr_doc_IDs_ordered,query_ids[i],true_docs,k)
			meanPrecision = meanPrecision + cp
		meanPrecision = meanPrecision/len(query_ids)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		s = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				s = s + 1
		recall = float(s/ len(true_doc_IDs))
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = 0.0

		for i in range(len(query_ids)):
			true_docs = []
			curr_doc_IDs_ordered = doc_IDs_ordered[i]
			for items in qrels:
				if query_ids[i] == int(items['query_num']):
					true_docs.append(int(items['id']))
			cp = self.queryRecall(curr_doc_IDs_ordered, query_ids[i], true_docs, k)
			meanRecall = meanRecall + cp
		meanRecall = meanRecall / len(query_ids)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if precision == 0 and recall == 0:
			return 0
		fscore = 2*precision*recall/(precision+recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""
		meanfscore = 0.0

		for i in range(len(query_ids)):
			true_docs = []
			curr_doc_IDs_ordered = doc_IDs_ordered[i]
			for items in qrels:
				if query_ids[i] == int(items['query_num']):
					true_docs.append(int(items['id']))
			cp = self.queryFscore(curr_doc_IDs_ordered, query_ids[i], true_docs, k)
			meanfscore = meanfscore + cp
		meanfscore = meanfscore / len(query_ids)

		return meanfscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
		dcg = 0.0
		idcg = 0.0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs.keys():
				val = true_doc_IDs[query_doc_IDs_ordered[i]]
				dcg = dcg + val/math.log2(i+2)
		nrels = sorted(true_doc_IDs.items(), key=lambda x:x[1],reverse=True)
		irels = dict(nrels)
		m = 0
		for j in irels.keys():
			idcg = idcg + irels[j]/math.log2(m+2)
			m = m + 1
			if m == k:
				break
		if idcg == 0:
			return 0
		nDCG = dcg/idcg
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""
		meanNDCG = 0.0
		for i in range(len(query_ids)):
			rel = {}
			curr_doc_IDs_ordered = doc_IDs_ordered[i]
			for items in qrels:
				if query_ids[i] == int(items['query_num']):
					rel[int(items['id'])] = 5 - int(items['position'])
			meanNDCG = meanNDCG + self.queryNDCG(curr_doc_IDs_ordered, query_ids[i],rel,k)
		meanNDCG = meanNDCG/len(query_ids)
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""
		curr = 0
		ans = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				curr = curr + 1
				ans = ans + (curr / (i + 1))
		if curr == 0:
			return 0
		avgPrecision = ans/curr
		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		map = 0.0

		for i in range(len(query_ids)):
			true_docs = []
			curr_doc_IDs_ordered = doc_IDs_ordered[i]
			for items in q_rels:
				if query_ids[i] == int(items['query_num']):
					true_docs.append(int(items['id']))
			cp = self.queryAveragePrecision(curr_doc_IDs_ordered, query_ids[i], true_docs, k)
			map = map + cp
		map = map / len(query_ids)

		return map

