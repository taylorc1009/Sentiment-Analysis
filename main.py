from keras.datasets import imdb
from six import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from IPython.display import Image
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

# inverts the { word: ID } dictionary order ('get_word_index()' returns a list of words with IDs
# see 'get_popular_ngrams()'
index = dict([(value, key) for (key, value) in imdb.get_word_index().items()])

vectorizer = CountVectorizer()

# this prevents the inclusion of specified strings, add any other strings you may want to prevent
occlusions = ['br', '#']  # for now, we only exclude system chars


def decode_sentence(sentence, occlude):  # converts a sentence from a list of integers (word IDs) to words
    decoded_sentence = ""
    for wordID in sentence:
        word = index.get(wordID - 3, "#")
        if not (occlude and word in occlusions):  # if occlude is true, we will remove words that are listed in occlude
            decoded_sentence += " " + word
    return decoded_sentence


def occlude_stopwords():  # adds stopwords (pronouns; words without much meaning to the sentiment) to the occlusions
    for o in stopwords.words('english'):
        occlusions.append(o)


def extract_data(data, occlude=False, tweet_tokenize=False):
    unigrams = {}
    bigrams = {}
    decoded = []

    if occlude:
        occlude_stopwords()

    for sentence in data:
        decoded_sentence = decode_sentence(sentence, occlude)

        # this allows differentiation of tokenizing as a Tweet - TweetTokenizer doesn't tokenize words by characters
        # like an apostrophe or a hyphen, i.e. we don't get unigrams such as "it" and "'s", we now get "it's"
        decoded_tokenized = TweetTokenizer.tokenize(TweetTokenizer(), decoded_sentence) if tweet_tokenize else \
            word_tokenize(decoded_sentence)

        for word in decoded_tokenized:  # unigrams
            # if not (occlude and word in occlusions):
            if not (occlude and tweet_tokenize and word[0] == '\''):
                if word not in unigrams:
                    unigrams[word] = 1
                else:
                    unigrams[word] += 1

        vectorizer.fit_transform(decoded_tokenized)

        grams = ngrams(decoded_tokenized, 2)  # bigrams
        for gram in grams:
            # if not (occlude and gram[0] in occlusions and gram[1] in occlusions):
            if not (occlude and tweet_tokenize and (gram[0] == '\'' or gram[1] == '\'')):
                if gram not in bigrams:
                    bigrams[gram] = 1
                else:
                    bigrams[gram] += 1

        decoded.append(decoded_sentence)

    print_popular_ngrams(unigrams, bigrams)

    return vectorizer.transform(sentence for sentence in decoded).toarray()


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


def analyse_data(data, input_dimension=max(len(testing_data), len(testing_targets))):
    tree = DecisionTreeClassifier(criterion='entropy').fit(data, training_targets)

    # TODO it may give a higher accuracy to have a bigger test set, but for some reason the tree complains that there's
    #  an inconsistency in the data dimensions -- EDIT: error has stopped, but for good?
    data_range = list(testing_data[i] for i in [j for j in range(input_dimension)])
    targets_range = list(testing_targets[i] for i in [j for j in range(input_dimension)])

    print("\nThe prediction accuracy is: ",
          tree.score(vectorizer.transform(decode_sentence(sentence, False) for sentence in
                                          data_range).toarray(),
                     targets_range) * 100, "%")

    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())  # TODO GraphViz appears to be missing the ability to display PNGs


vectorized_data = extract_data(training_data, occlude=False, tweet_tokenize=False)
analyse_data(vectorized_data, input_dimension=5000)
