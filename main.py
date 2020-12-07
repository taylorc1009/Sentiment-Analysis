from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# splits the data into training and testing data, the target data signifies whether or not each sentence will be used
# for training/testing TODO (why?)
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()

# inverts the { word: ID } dictionary order ('get_word_index()' returns a list of words with IDs
# see 'get_popular_ngrams()'
index = dict([(value, key) for (key, value) in imdb.get_word_index().items()])

vectorizer = CountVectorizer()

# this prevents the inclusion of pronouns; words without much meaning to the sentiment
occlusions = stopwords.words('english')
for o in ['br', '#', '\'']:  # add any other strings you may want to prevent analysing
    occlusions.append(o)


def decode_sentence(sentence, occlude):  # converts a sentence from a list of integers (word IDs) to words
    decoded_sentence = ""
    for wordID in sentence:
        word = index.get(wordID - 3, "#")
        if not (occlude and word in occlusions):  # if occlude is true, we will remove words that are listed in occlude
            decoded_sentence += " " + word
    return decoded_sentence


def extract_data(data, occlude):
    unigrams = {}
    bigrams = {}
    decoded = []

    for sentence in data:
        decoded_sentence = decode_sentence(sentence, occlude)
        decoded_tokenized = TweetTokenizer.tokenize(TweetTokenizer(), decoded_sentence)

        for word in decoded_tokenized:  # unigrams
            if not (occlude and word in occlusions):
                if word not in unigrams:
                    unigrams[word] = 1
                else:
                    unigrams[word] += 1

        vectorizer.fit_transform(decoded_tokenized)

        grams = ngrams(decoded_tokenized, 2)  # bigrams
        for gram in grams:
            if not (occlude and gram[0] in occlusions and gram[1] in occlusions):
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


vectorized_data = extract_data(training_data, False)

tree = DecisionTreeClassifier(criterion='entropy').fit(vectorized_data, training_targets)

print("\nThe prediction accuracy is: ", tree.score(vectorizer.transform(decode_sentence(sentence, False) for sentence in
                                                                        testing_data).toarray(),
                                                   testing_targets) * 100, "%")
