# 4.2.1 The Reuters dataset
# Listing 4.11 Loading the Reuters dataset
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(f"len(train_data): {len(train_data)}")
print(f"len(test_data): {len(test_data)}")

# Listing 4.12 Decoding newswires back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join(
        [reverse_word_index.get(i - 3, "?") for i in train_data[0]])
print(decoded_newswire)

#decoded_newswire = " ".join(
#        [reverse_word_index.get(i - 3, "?") for i in test_data[1]])
#print(decoded_newswire)

print(f"train_labels[10]: {train_labels[10]}")

# 4.2.2 Preparing data
# Listing 4.3 Encoding the integer sequences via multi-hot encoding
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
            return results

# Listing 4.13 Encoding the input data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Listing 4.14 Encoding the labels
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

print(y_train.shape)
print(y_test.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print(y_train.shape)
print(y_test.shape)

# 4.2.3 Building your model


