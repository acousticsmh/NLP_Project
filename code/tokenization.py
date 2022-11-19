from util import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
# Add your import statements here


class Tokenization():

    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []
        for i in text:
            l1 = split_into_words(i)
            l2 = [j.lstrip('0123456789.- ') for j in l1]
            l3 = [i for i in l2 if (i != ',' and i != '.' and i != '' and i != ' ' and len(i) >= 3)]
            tokenizedText.append(l3)
        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """
        tokenizedText = []
        tokenizer = TreebankWordTokenizer()
        for i in text:
            l1 = tokenizer.tokenize(i)
            l2 = [i for i in l1 if (i != ',' and i != '.')]
            tokenizedText.append(l2)
        return tokenizedText


    def regtok(self,text):
        tk = RegexpTokenizer('\s+',gaps=True)
        tokenizedText = []
        for i in text:
            l1 = tk.tokenize(i)
            l2 = [i for i in l1 if (i != ',' and i != '.')]
            tokenizedText.append(l2)
        return tokenizedText