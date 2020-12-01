from keras.datasets import imdb
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in training_data[0]] )
print(decoded)