import re
from nltk.corpus import wordnet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import math
import numpy.linalg as LA


special_characters = ["\'\'", "\"\"", "!", "\"", "#", "$", "%", "&",
                      "(", ")", "-""*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^", "`", "{", "|", "}", "~", "'",",","."]
sentence_break = [".","!"]

vocab = set()


def listToString(s):
    """
		Convert a list of words into a space-separated string

		Parameters
		----------
		arg1 : list
			A list of words (strings)
		Returns
		-------
		string : A space-separated string of all words in list
	"""
    str1 = ""
    for ele in s:
        str1 = str1 + " " + ele
    return str1


def cx(a, b):
    """
		Cosine of Angle between two vectors
		Parameters
		----------
		arg1 : array
			First vector
		arg2 : array
			Second vector
		Returns
		-------
		float : cosine of angle between two vectors
	"""
    ans = round((np.inner(a,b)/LA.norm(a)*LA.norm(b)),5)
    return ans




def split_into_sentences(text):
    """
		Naive sentence Segmentor

		Parameters
		----------
		arg1 : list
			A text containing the content of a document
		Returns
		-------
		list : A list of lists where each sub-list is a sentence.
	"""
    sentences = re.split('!.',text)
    sentences = [s.strip() for s in sentences]
    sentences = list(filter(None, sentences))
    return sentences


def split_into_words(text):
	"""
		Split a sentence into words (Naive implementation)

		Parameters
		----------
		arg1 : string
		A sentence of a document in string form
		Returns
		-------
		list : A list of tokens in that sentence.
	"""
	for i in special_characters:
		text = text.replace(i,' ')
	words = text.split()
	return words


def get_wordnet_pos(treebank_tag):
	"""
		Returns the wordnet tag of a word

		Parameters
		----------
		arg1 : treebank_tag
		Returns
		-------
		Part of speech for that tag
	"""

	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN


def IDF(N,n):
	"""
		Returns the IDF score for a word occuring in n of N documents

		Parameters
		----------
		arg1 : Total number of documents in corpus
		arg2 : Number of documents containing a term q
		Returns
		-------
		float : IDF score
	"""
	a = (N-n+0.5)/(n+0.5)
	a = a + 1
	b = np.log(a)
	return b

def fun(f,k,b,le,avg):
    a = f*(k+1)/(f + k*(1-b + b*(le/avg)))
    return a