from util import *
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
# Add your import statements here


class StopwordRemoval():

    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence with stopwords removed
        """

        stopwordRemovedText = []

        for i in text:
            l = [j for j in i if not j in stop_words]
            vocab.update(l)
            stopwordRemovedText.extend(l)
        return stopwordRemovedText
