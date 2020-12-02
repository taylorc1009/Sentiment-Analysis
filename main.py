from keras.datasets import imdb
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
import nltk
nltk.download('punkt')
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()

# index = imdb.get_word_index()
# reverse_index = dict([(value, key) for (key, value) in index.items()])
# decoded = " ".join([reverse_index.get(i - 3, "#") for i in training_data[0]])
# print(decoded)


def get_popular_ngrams(sentences):
    unigrams, bigrams = extract_ngrams(sentences)
    popular_unigrams, popular_bigrams = Counter(unigrams).most_common(20), Counter(bigrams).most_common(20)

    i = 0
    print("Top 20 unigrams")
    for unigram in popular_unigrams:
        i += 1
        print(str(i) + ". " + unigram[0] + " : " + str(unigram[1]))

    i = 0
    print("\nTop 20 bigrams")
    for bigram in popular_bigrams:
        i += 1
        print(str(i) + ". " + "{ " + bigram[0][0] + ", " + bigrams[0][1] + " } : " + str(bigram[1]))


def extract_ngrams(sentences):
    unigrams = []
    bigrams = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)

        temp = extract_unigrams(tokens)
        if temp:
            for u in temp:
                unigrams.append(u)

        temp = extract_bigrams(tokens)
        if temp:
            for b in temp:
                bigrams.append(b)

    return unigrams, bigrams


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


get_popular_ngrams(imdb.get_word_index())
