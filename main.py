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

# inverts the { word: ID } dictionary order
index = dict([(value, key) for (key, value) in imdb.get_word_index().items()])


def get_popular_ngrams(sentences):
    decoded = []
    occlude = stopwords.words('english')

    # TODO are these needed for better sentiment analysis? it would appear that they are (for example, there's a
    # TODO difference between "could" and "couldn't") but it would be better if the tokenizer would not split these
    for o in ['br', '#', '\'', '\'ve', 'n\'t', '\'s']:
        occlude.append(o)
    for sentence in sentences:
        decoded.append(" ".join([index.get(wordID - 3, "#") for wordID in sentence]))
    unigrams, bigrams = extract_ngrams(decoded, occlude)

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
    for sentence in sentences:
        # we use TweetTokenizer as it doesn't split words by apostrophe - TODO is this better?
        words = nltk.TweetTokenizer().tokenize(sentence)

        for word in words:
            if word not in occlude:
                if word not in unigrams:
                    unigrams[word] = 1
                else:
                    unigrams[word] += 1

        grams = ngrams(words, 2)
        for gram in grams:
            if gram[0] not in occlude and gram[1] not in occlude:
                if gram not in bigrams:
                    bigrams[gram] = 1
                else:
                    bigrams[gram] += 1

    return unigrams, bigrams


# print(decoded)

get_popular_ngrams(training_data)

# vectorizer = CountVectorizer()
# imdb_bag_of_words = vectorizer.fit_transform(imdb.get_word_index())
# vectorizer.get_feature_names()
# bow_vector = imdb_bag_of_words.toarray()
# vectorizer.vocabulary_.get('awesome')
# vectorizer.transform(['Something completely new.']).toarray()

# tokenizer = Tokenizer(imdb_bag_of_words.size())
