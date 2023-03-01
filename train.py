import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM
from tensorflow.keras.datasets import imdb


num_words = 10000 # nombre de mots à utiliser
maxlen = 100 # longueur maximale de chaque avis
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)


x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


model = keras.models.Sequential([
    Embedding(num_words, 32),
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)