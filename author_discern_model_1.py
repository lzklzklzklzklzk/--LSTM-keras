from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Dense, Flatten, LSTM, GRU, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

# extracting train data
train_path = './train.txt'

train_texts = []
train_labels = []

with open(train_path, encoding='utf-8') as f_train:
    for line in f_train:
        train_labels.append(line[-2:-1])
        train_texts.append(line[:-7])

# extracting test data
test_path = './self_test.txt'

test_texts = []
test_labels = []

with open(test_path, encoding='utf-8') as f_test:
    for line in f_test:
        test_labels.append(line[-2:-1])
        test_texts.append(line[:-7])

max_words = 20000
maxlen = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
word_index = tokenizer.index_word

data = pad_sequences(sequences, maxlen=maxlen)
train_labels = np.array(train_labels)

one_hot_train_labels = to_categorical(train_labels)

embedding_dim = 256
# setup the model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['acc'])

# training
history = model.fit(data, one_hot_train_labels, epochs=12,
                    batch_size=32, validation_split=0.2)

model.save_weights('model_1.h5')

# plot the training process
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# processing the test data
sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(sequences, maxlen=maxlen)
test_labels = np.array(test_labels, dtype=np.int32)

model.load_weights('model_1.h5')
test_pred = model.predict(test_data)
test_pred = np.argmax(test_pred, axis=1)

# test the model on the teset data
print('micro precision: %f' % precision_score(
    test_labels, test_pred, average='micro'))
print('micro recall: %f' % recall_score(
    test_labels, test_pred, average='micro'))
print('micro f1 score: %f' % f1_score(
    test_labels, test_pred, average='micro'))

print('macro precision: %f' % precision_score(
    test_labels, test_pred, average='macro'))
print('macro recall: %f' % recall_score(
    test_labels, test_pred, average='macro'))
print('macro f1 score: %f' % f1_score(
    test_labels, test_pred, average='macro'))

# generating the results on test.txt
# extracting final test data
test_path = './test.txt'

test_texts = []
test_labels = []

with open(test_path, encoding='utf-8') as f_test:
    for line in f_test:
        test_texts.append(line[:])
sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(sequences, maxlen=maxlen)
test_labels = np.array(test_labels, dtype=np.int32)

model.load_weights('model_1.h5')
test_pred = model.predict(test_data)
test_pred = np.argmax(test_pred, axis=1)

result_path = './results.txt'

with open(result_path, 'w') as f_results:
    for i in test_pred:
        f_results.write(str(i) + '\n')

print(test_pred.shape)
