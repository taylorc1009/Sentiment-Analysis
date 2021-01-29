from sklearn import metrics, logger
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

nltk.download('punkt')

# splits the data into training and testing data, the target data signifies whether or not each sentence will be used
# for training/testing (target determines the sentiment of a review - 1 = negative, 2 = positive)
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()

# Application settings
occlude_stopwords = True  # whether or not to prevent words without much sentiment from being used
use_tweet_tokenizer = True  # use TweetTokenizer to split the decoded sentence; gives cleaner data (see 'extract_data')
training_dim = 5000  # n values to provide for training (max amount results in an extremely large amount of operations)
testing_dim = 1000  # amount of values to test with; if you want all, use = max(len(testing_data), len(testing_targets))
use_bigrams = False  # should the bigrams be used as the bag of words vocabulary?
export_tree_model = False  # would you like a graphic of the tree model?

# inverts the { word: ID } dictionary order ('get_word_index()' returns a list of words with IDs)
index = dict([(value, key) for (key, value) in imdb.get_word_index().items()])

vectorizer = CountVectorizer()

# this prevents the inclusion of specified strings, add any other strings you may want to prevent
system_occlusions = ['br', '#']  # for now, we only exclude system chars
if occlude_stopwords:
    nltk.download('stopwords')
    occlusions = stopwords.words('english')


def decode_sentence(sentence):  # converts a sentence from a list of integers (word IDs) to words
    decoded_sentence = ""
    for wordID in sentence:  # converts every word ID in a sentence to their respective word from the index
        word = index.get(wordID - 3, "#")

        # if occlude is true, we will prevent any words that are listed in occlusions from being added
        if not (occlude_stopwords and word in occlusions):
            decoded_sentence += " " + word

    return decoded_sentence


# if we're not training or testing (both booleans are 'False'), we can still use this to get the data without
def extract_data(data, is_train=False, is_test=False):
    unigrams_ = {}
    bigrams_ = {}
    sentence_vocabularies = []
    concatenated_bigrams_str = ""
    bags_of_words_ = []

    try:
        if is_train and is_test:
            raise ValueError('\'is_train\' and \'is_test\' were true; we can\'t train with test data and vice-versa.')
        for sentence in data:
            decoded_sentence = decode_sentence(sentence)

            # this allows differentiation of tokenizing as a Tweet - TweetTokenizer doesn't tokenize words by characters
            # like an apostrophe or a hyphen, i.e. we don't get unigrams such as "it" and "'s", we now get "it's"
            decoded_tokenized = TweetTokenizer.tokenize(TweetTokenizer(), decoded_sentence) if use_tweet_tokenizer \
                else word_tokenize(decoded_sentence)

            if not is_test:  # we won't need unigrams for testing purposes
                unigrams_ = extract_unigrams(unigrams_, decoded_tokenized)

            # during testing, we only need bigrams if we want a vocabulary of bigrams
            if not is_test or (is_test and use_bigrams):
                # concatenated_bigrams - used to add the bigrams to the vocabulary
                # concatenated_bigrams_str - used to get the bigrams in the bag of words
                bigrams_, concatenated_bigrams_str, concatenated_bigrams = extract_ngrams(bigrams_, decoded_tokenized)

            if use_bigrams:  # builds sentences using either unigrams or bigrams
                sentence_vocabularies.append(concatenated_bigrams_str)
            else:
                sentence_vocabularies.append(decoded_sentence[3:])  # start at index 3 as 0-2 = ' # '

        if is_train:  # initialise the vocabulary with the unique words in training data
            vectorizer.fit(unigrams_.keys())

        # vectorizer.transform creates a bag of words (as count vectors) - a list of the count of each word of the
        # vocabulary in the sentence (0 if a word isn't in the sentence)
        bags_of_words_ = vectorizer.transform(vocabulary for vocabulary in sentence_vocabularies).toarray()
    except Exception as e:
        logger.error(repr(e))

    return unigrams_, bigrams_, bags_of_words_


def extract_unigrams(unigrams_, tokenized_sentence):
    for word in tokenized_sentence:
        # prevents stand-alone apostrophes from being used as unigrams
        if not (word in system_occlusions or (use_tweet_tokenizer and word[0] == '\'')):
            if word not in unigrams_:  # counts the occurrence of each unique unigram
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
        if not ((gram[0] in system_occlusions or gram[1] in system_occlusions) or (use_tweet_tokenizer and (gram[0] == "'" or gram[1] == "'"))):
            if gram not in ngrams_:  # counts the occurrence of each unique bigram
                ngrams_[gram] = 1

                if use_bigrams:
                    concatenated_gram = ' '.join(word for word in gram)
                    concatenated_grams_str += ' ' + concatenated_gram
                    concatenated_grams.append(concatenated_gram)
            else:
                ngrams_[gram] += 1
    return ngrams_, concatenated_grams_str, concatenated_grams


def print_popular_ngrams(unigrams_, bigrams_):  # sorts the unigrams and bigrams, then displays the 20 most occurring
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


def analyse_data(training_bag_of_words, depth=5):
    # trains the tree using the training data as bags of words with their respective labels (sentiments)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth).fit(training_bag_of_words, training_targets[:training_dim])

    # gets only the testing data bag of words from the extract
    _, _, data_in_range = extract_data(testing_data[:testing_dim], is_test=True)
    targets_in_range = testing_targets[:testing_dim]

    print('\n=== Metrics ===')
    print('\nPrediction Accuracy: ', tree.score(data_in_range, targets_in_range) * 100, "%")
    print('\nRecall: ', metrics.recall_score(targets_in_range, tree.predict(data_in_range)) * 100, "%")
    print('\nConfusion matrix:\n', metrics.confusion_matrix(targets_in_range, tree.predict(data_in_range)),
          '\nMatrix Column and Row labels (both are in the same order):\n 1 = False\n 2 = True')

    export_tree(tree)


def export_tree(tree):  # outputs a GraphViz tree object as an SVG
    if export_tree_model:
        dot_data = StringIO()
        export_graphviz(tree, out_file=dot_data, filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write(path='tree.svg', format='svg')


unigrams, bigrams, bags_of_words = extract_data(training_data[:training_dim], is_train=True)
print_popular_ngrams(unigrams, bigrams)
analyse_data(bags_of_words)
