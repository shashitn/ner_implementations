import codecs
import collections
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def preprocess_data(dataset_filepath):
    print("Preprocessing data..............")

    def _parse_dataset(dataset_filepath):
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)

        line_count = -1
        tokens = []
        labels = []
        new_token_sequence = []
        new_label_sequence = []
        if dataset_filepath:
            f = codecs.open(dataset_filepath, 'r', 'UTF-8')
            for line in f:
                line_count += 1
                line = line.strip().split(' ')
                if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
                    if len(new_token_sequence) > 0:
                        labels.append(new_label_sequence)
                        tokens.append(new_token_sequence)
                        new_token_sequence = []
                        new_label_sequence = []
                    continue
                token = str(line[0])
                label = str(line[-1])
                token_count[token] += 1
                label_count[label] += 1

                new_token_sequence.append(token)
                new_label_sequence.append(label)

                for character in token:
                    character_count[character] += 1

                # if self.debug and line_count > 200: break# for debugging purposes

            if len(new_token_sequence) > 0:
                labels.append(new_label_sequence)
                tokens.append(new_token_sequence)
            f.close()
        return labels, tokens, token_count, label_count, character_count

    labels, tokens, token_count, label_count, character_count = _parse_dataset('/content/ner_final.eng')

    token_list = list((set([y for x in tokens for y in x])))
    label_list = list(set([y for x in labels for y in x]))

    word2ind = {word: index for index, word in enumerate(token_list)}
    ind2word = {index: word for index, word in enumerate(token_list)}
    label_unique = list(set([c for x in label_list for c in x]))
    label2ind = {label: (index + 1) for index, label in enumerate(label_list)}
    ind2label = {(index + 1): label for index, label in enumerate(label_list)}

    def encode(x, n):
        result = np.zeros(n)
        result[x] = 1
        return result

    X = tokens
    y = labels

    maxlen = max([len(x) for x in X])
    print('Maximum sequence length:', maxlen)

    X_enc = [[word2ind[c] for c in x] for x in X]
    max_label = max(label2ind.values()) + 1
    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

    X_enc = pad_sequences(X_enc, maxlen=maxlen)
    y_enc = pad_sequences(y_enc, maxlen=maxlen)

    X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=11 * 32, train_size=45 * 32,
                                                        random_state=42)
    print('Training and testing tensor shapes:')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, maxlen, word2ind, label2ind
