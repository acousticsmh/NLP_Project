from util import *
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()
# Add your import statements here


class InflectionReduction:

    def reduce(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of
                stemmed/lemmatized tokens representing a sentence
        """

        reducedText = []

        for i in text:
            temp = []
            tags = nltk.pos_tag(i)
            for j in i:
                pos = [a_tuple[1] for (index, a_tuple)
                       in enumerate(tags) if a_tuple[0] == j]
                temp.append(lemmatizer.lemmatize(j, get_wordnet_pos(pos[0])))
            reducedText.append(temp)
        return reducedText
