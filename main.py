from sklearn import metrics
from keras.datasets import imdb
from six import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.tree import export_graphviz
import pydotplus
import nltk
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'
nltk.download('punkt')
nltk.download('stopwords')

# splits the data into training and testing data, the target data signifies whether or not each sentence will be used
# for training/testing (target determines the sentiment of a review - 1 = negative, 2 = positive)
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()

# Application settings
occlude_stopwords = True  # whether or not to prevent words without much sentiment from being used
use_tweet_tokenizer = True  # use TweetTokenizer to split the decoded sentence; gives cleaner data (see 'extract_data')
training_dim = 5000  # n values to provide for training (max amount results in an extremely large amount of operations)
testing_dim = 1000  # amount of values to test with; if you want all, use = max(len(testing_data), len(testing_targets))
use_bigrams = False  # should the bigrams be used as the bag of words vocabulary?

# inverts the { word: ID } dictionary order ('get_word_index()' returns a list of words with IDs
# see 'get_popular_ngrams()'
index = dict([(value, key) for (key, value) in imdb.get_word_index().items()])

vectorizer = CountVectorizer()

# this prevents the inclusion of specified strings, add any other strings you may want to prevent
system_occlusions = ['br', '#']  # for now, we only exclude system chars
occlusions = stopwords.words('english')


def decode_sentence(sentence):  # converts a sentence from a list of integers (word IDs) to words
    decoded_sentence = ""
    for wordID in sentence:
        word = index.get(wordID - 3, "#")

        # if occlude is true, we will prevent any words that are listed in occlusions from being added
        if not (occlude_stopwords and word in occlusions):
            decoded_sentence += " " + word

    return decoded_sentence


def extract_data(data, is_test=False):
    unigrams_ = {}
    bigrams_ = {}
    sentence_vocabularies = []
    concatenated_bigrams_str = ""

    for sentence in data:
        decoded_sentence = decode_sentence(sentence)

        # this allows differentiation of tokenizing as a Tweet - TweetTokenizer doesn't tokenize words by characters
        # like an apostrophe or a hyphen, i.e. we don't get unigrams such as "it" and "'s", we now get "it's"
        decoded_tokenized = TweetTokenizer.tokenize(TweetTokenizer(), decoded_sentence) if use_tweet_tokenizer else \
            word_tokenize(decoded_sentence)

        if not is_test:  # we won't need unigrams for testing purposes
            unigrams_ = extract_unigrams(unigrams_, decoded_tokenized)

        if not is_test or (is_test and use_bigrams):
            # concatenated_bigrams - used to add the bigrams to the vocabulary
            # concatenated_bigrams_str - used to get the bigrams in the bag of words
            bigrams_, concatenated_bigrams_str, concatenated_bigrams = extract_ngrams(bigrams_, decoded_tokenized)

        if use_bigrams:
            sentence_vocabularies.append(concatenated_bigrams_str)
        else:
            sentence_vocabularies.append(decoded_sentence[3:])

    if not is_test:
        vectorizer.fit(unigrams_.keys())

    return unigrams_, bigrams_, vectorizer.transform(vocabulary for vocabulary in sentence_vocabularies).toarray()


def extract_unigrams(unigrams_, tokenized_sentence):
    for word in tokenized_sentence:
        # prevents stand-alone apostrophes from being used as tokens
        if not (word in system_occlusions or (use_tweet_tokenizer and word[0] == '\'')):
            if word not in unigrams_:
                unigrams_[word] = 1
            else:
                unigrams_[word] += 1
    return unigrams_


def extract_ngrams(ngrams_, tokenized_sentence, n=2):
    concatenated_grams = []
    concatenated_grams_str = ''

    grams = ngrams(tokenized_sentence, n)
    for gram in grams:
        # this if prevents system characters from being used, such as 'br', and the tweet tokenizer also gives
        # apostrophes as their own token so this prevents that also
        if not ((gram[0] in system_occlusions or gram[1] in system_occlusions) or (use_tweet_tokenizer and (gram[0] ==
                                                                                                            "'" or gram[
                                                                                                                1] ==
                                                                                                            "'"))):
            if gram not in ngrams_:
                ngrams_[gram] = 1

                if use_bigrams:
                    concatenated_gram = ' '.join(word for word in gram)
                    concatenated_grams_str += ' ' + concatenated_gram
                    concatenated_grams.append(concatenated_gram)
            else:
                ngrams_[gram] += 1
    return ngrams_, concatenated_grams_str, concatenated_grams


def print_popular_ngrams(unigrams_, bigrams_):
    i = 0
    print("\nTop 20 unigrams")
    for unigram in dict(sorted(unigrams_.items(), key=lambda item: item[1], reverse=True)):
        i += 1
        if i > 20:
            break
        print(str(i) + ".", unigram, "::", unigrams_[unigram])

    i = 0
    print("\nTop 20 bigrams")
    for bigram in dict(sorted(bigrams_.items(), key=lambda item: item[1], reverse=True)):
        i += 1
        if i > 20:
            break
        print(str(i) + ".", bigram, "::", bigrams_[bigram])


def analyse_data(training_bag_of_words):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=5).fit(training_bag_of_words,
                                                                        training_targets[:training_dim])

    data_in_range = extract_data(testing_data[:testing_dim], is_test=True)[2]
    targets_in_range = testing_targets[:testing_dim]
    print("\nPrediction Accuracy: ", tree.score(data_in_range, targets_in_range) * 100, "%")
    print("\nRecall: ", metrics.recall_score(targets_in_range, tree.predict(data_in_range)) * 100, "%")

    # outputs a GraphViz tree object as an SVG
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write(path='tree.svg', format='svg')


unigrams, bigrams, bags_of_words = extract_data(training_data[:training_dim])
print_popular_ngrams(unigrams, bigrams)
analyse_data(bags_of_words)
