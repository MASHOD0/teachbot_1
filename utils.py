import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer


def tokenize(sentence):
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

def stem(word):
    """
    stemming = find the root form of the word
    :param word: word to stem
    :return: stemmed word
    """
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    
    """

    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag