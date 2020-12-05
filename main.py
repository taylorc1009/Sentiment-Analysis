from keras.datasets import imdb
from keras_preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# splits the data into training and testing data, the target data signifies whether or not each sentence will be used
# for training/testing TODO (why?)
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()

# inverts the { word: ID } dictionary order ('get_word_index()' returns a list of words with IDs
# see 'get_popular_ngrams()'
index = dict([(value, key) for (key, value) in imdb.get_word_index().items()])


def get_popular_ngrams(sentences):
    # this prevents the inclusion of pronouns; words without much meaning to the sentiment
    occlude = stopwords.words('english')
    for o in ['br', '#']:
        occlude.append(o)

    unigrams, bigrams, decoded = extract_ngrams(sentences, occlude)

    i = 0
    print("Top 20 unigrams")
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


def extract_ngrams(sentences, occlude):
    unigrams = {}
    bigrams = {}
    decoded = []

    # this doesn't tokenize to get the unigrams because it gets the unigrams as well as decoding the sentence at the
    # same time, instead of decoding by iterating every ID in each sentence then iterating every decoded word;
    # essentially, they're now merged
    for sentence in sentences:
        decoded_sentence = ""

        for wordID in sentence:
            word = index.get(wordID - 3, "#")
            decoded_sentence += " " + word
            if word not in occlude:
                if word not in unigrams:
                    unigrams[word] = 1
                else:
                    unigrams[word] += 1

        grams = ngrams(decoded_sentence.split(' '), 2)
        for gram in grams:
            if gram[0] not in occlude and gram[1] not in occlude:
                if gram not in bigrams:
                    bigrams[gram] = 1
                else:
                    bigrams[gram] += 1

        decoded.append(decoded_sentence)

    return unigrams, bigrams, decoded


get_popular_ngrams(training_data)

# vectorizer = CountVectorizer()
# imdb_bag_of_words = vectorizer.fit_transform(imdb.get_word_index())
# vectorizer.get_feature_names()
# bow_vector = imdb_bag_of_words.toarray()
# vectorizer.vocabulary_.get('awesome')
# vectorizer.transform(['Something completely new.']).toarray()

# tokenizer = Tokenizer(imdb_bag_of_words.size())
