from keras.layers import Embedding, Bidirectional, TimeDistributed, LSTM, Dense, Activation
from keras.models import Sequential

import data_prep as prep
import ner_utils as utils


def lstm_model(max_features, embedding_size, maxlen, hidden_size, out_size):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_size,
                        input_length=maxlen, mask_zero=True))
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
    model.add(TimeDistributed(Dense(out_size)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def __main__():
    save_path = prep.read_data()
    file_path = prep.prep_data(save_path)
    X_train, X_test, y_train, \
    y_test, max_len, word2ind, label2ind = utils.preprocess_data(file_path)

    # Hyperparameters
    max_features = len(word2ind)
    embedding_size = 128
    hidden_size = 32
    out_size = len(label2ind) + 1
    batch_size = 32
    epochs = 40

    model = lstm_model(max_features, embedding_size, max_len, hidden_size, out_size)
    print(model.summary())

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_train, y_test))

    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Raw test score:', score)


__main__()
