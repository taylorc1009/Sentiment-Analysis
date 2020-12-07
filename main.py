from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.densenet import layers

nltk.download('punkt')
nltk.download('stopwords')

# splits the data into training and testing data, the target data signifies whether or not each sentence will be used
# for training/testing TODO (why?)
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()

# inverts the { word: ID } dictionary order ('get_word_index()' returns a list of words with IDs
# see 'get_popular_ngrams()'
index = dict([(value, key) for (key, value) in imdb.get_word_index().items()])

vectorizer = CountVectorizer()


def print_popular_ngrams(unigrams, bigrams):
    i = 0
    print("\nTop 20 unigrams")
    for unigram in dict(sorted(unigrams.items(), key=lambda item: item[1], reverse=True)):
        i += 1
        if i > 20:
            break
        print(str(i) + ".", unigram, "::", unigrams[unigram])

    i = 0
    print("\nTop 20 bigrams")
    for bigram in dict(sorted(bigrams.items(), key=lambda item: item[1], reverse=True)):
        i += 1
        if i > 20:
            break
        print(str(i) + ".", bigram, "::", bigrams[bigram])


def extract_data(data):
    unigrams = {}
    bigrams = {}
    decoded = []

    # this prevents the inclusion of pronouns; words without much meaning to the sentiment
    occlude = []  # stopwords.words('english')
    for o in ['br', '#']:
        occlude.append(o)

    for sentence in data:
        decoded_sentence = decode_sentence(sentence)
        decoded_tokenized = word_tokenize(decoded_sentence)

        for word in decoded_tokenized:  # unigrams
            if word not in occlude:
                if word not in unigrams:
                    unigrams[word] = 1
                else:
                    unigrams[word] += 1

        vectorizer.fit_transform(decoded_tokenized)

        grams = ngrams(decoded_tokenized, 2)  # bigrams
        for gram in grams:
            if gram[0] not in occlude and gram[1] not in occlude:
                if gram not in bigrams:
                    bigrams[gram] = 1
                else:
                    bigrams[gram] += 1

        decoded.append(decoded_sentence)

    print_popular_ngrams(unigrams, bigrams)

    return vectorizer.transform(sentence for sentence in decoded).toarray()


def decode_sentence(sentence):  # converts a sentence from a list of integers (word IDs) to words
    decoded_sentence = ""
    for wordID in sentence:
        word = index.get(wordID - 3, "#")
        decoded_sentence += " " + word
    return decoded_sentence


vectorized_data = extract_data(training_data)

tree = DecisionTreeClassifier(criterion='entropy').fit(vectorized_data, training_targets)

print("\nThe prediction accuracy is: ", tree.score(vectorizer.transform(decode_sentence(sentence) for sentence in
                                                                        testing_data).toarray(),
                                                   testing_targets) * 100, "%")
