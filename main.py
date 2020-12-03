import operator

from keras.datasets import imdb
from keras_preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import nltk

nltk.download('punkt')
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()


# index = imdb.get_word_index()
# reverse_index = dict([(value, key) for (key, value) in index.items()])
# decoded = " ".join([reverse_index.get(i - 3, "#") for i in training_data[0]])
# print(decoded)


def get_popular_ngrams(sentences):
    unigrams, bigrams = extract_ngrams(sentences)
    i = 0
    print("Top 20 unigrams")
    for unigram in dict(sorted(unigrams.items(), key=lambda item: item[1], reverse=True)):
        i += 1
        if i > 20:
            break
        print(str(i), ". ", unigram, " :: ", unigrams[unigram])

    i = 0
    print("\nTop 20 bigrams")
    for bigram in dict(sorted(bigrams.items(), key=lambda item: item[1], reverse=True)):
        i += 1
        if i > 20:
            break
        print(str(i), ". ", "{ ", bigram, " } : ", bigrams[bigram])


def extract_ngrams(sentences):
    unigrams = {}
    bigrams = {}
    for sentence in sentences:
        tokens = word_tokenize(sentence)

        for token in tokens:
            if token not in unigrams:
                unigrams[token] = 1
            else:
                unigrams[token] += 1

        grams = ngrams(tokens, 2)
        for gram in grams:
            if gram not in bigrams:
                bigrams[gram] = 1
            else:
                bigrams[gram] += 1

    return unigrams, bigrams


get_popular_ngrams(imdb.get_word_index())

# vectorizer = CountVectorizer()
# imdb_bag_of_words = vectorizer.fit_transform(imdb.get_word_index())
# vectorizer.get_feature_names()
# bow_vector = imdb_bag_of_words.toarray()
# vectorizer.vocabulary_.get('awesome')
# vectorizer.transform(['Something completely new.']).toarray()

# tokenizer = Tokenizer(imdb_bag_of_words.size())
