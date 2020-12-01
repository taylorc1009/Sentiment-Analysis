from keras.datasets import imdb
from nltk import word_tokenize
from nltk.util import ngrams
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()

# index = imdb.get_word_index()
# reverse_index = dict([(value, key) for (key, value) in index.items()])
# decoded = " ".join([reverse_index.get(i - 3, "#") for i in training_data[0]])
# print(decoded)


def extract_ngrams(sentences):
    unigrams = []
    bigrams = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        unigrams.append(extract_unigrams(tokens))
        bigrams.append(extract_bigrams(tokens))


def extract_unigrams(tokens):
    unigrams = []
    for token in tokens:
        if token not in unigrams:
            unigrams.append(token)
    return unigrams


def extract_bigrams(tokens):
    bigrams = []
    grams = ngrams(tokens, 2)
    for gram in grams:
        if gram not in bigrams:
            bigrams.append(gram)
    return bigrams
