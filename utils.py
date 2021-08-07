import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer


def q_tokenize(sentence):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.
    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    :param sentence: string to tokenize
    :return: tokenized list
    """
    tokens = nltk.word_tokenize(sentence)
    return tokens

