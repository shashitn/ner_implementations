import pandas as pd


def read_data(file_path='data/ner_dataset_utf8.csv', save_path='ner.json'):
    print("Reading data..............")
    df = pd.read_csv(file_path)
    df['Sentence #'] = df['Sentence #'].fillna("empty")
    sent_id = []
    for i in range(0, len(df)):
        if df['Sentence #'][i] != 'empty':
            sent_id.append(df['Sentence #'][i])
        else:
            df.at[i, 'Sentence #'] = sent_id[-1]

    for i in range(0, len(df)):
        df.at[i, 'sentence_id'] = df['Sentence #'][i].split(" ")[1]

    df.to_json(save_path, orient='records')
    print('ner.json stored @ ' + save_path)

    return save_path


def prep_data(file_path):
    print("Preparing data.........")
    df = pd.read_json(file_path)
    print("Here")
    for i in range(0, len(df)):
        if df['Word'][i] == '0':
            df.at[i, 'Word'] = '.'

    print("Buliding NER set")
    ner_set = []
    last_word = "ner_lstm"
    for i in range(0, len(df)):
        if last_word != 'ner_lstm':
            if last_word[-1] != '.':
                # print(df['Word'][i] + " " + df['POS'][i] + " " + df['Tag'][i])
                ner_set.append(df['Word'][i] + " " + df['POS'][i] + " " + df['Tag'][i])
                last_word = df['Word'][i]
            else:
                # print("")
                # print(df['Word'][i] + " " + df['POS'][i] + " " + df['Tag'][i])
                ner_set.append("")
                ner_set.append(df['Word'][i] + " " + df['POS'][i] + " " + df['Tag'][i])
                last_word = df['Word'][i]
        else:
            # print(df['Word'][i] + " " + df['POS'][i] + " " + df['Tag'][i])
            ner_set.append(df['Word'][i] + " " + df['POS'][i] + " " + df['Tag'][i])
            last_word = df['Word'][i]

        with open('ner_final.eng', mode="w") as outfile:
            for s in ner_set:
                outfile.write("%s\n" % s)
    print("Final set written to " + "/content/ner_final.eng")

    return 'ner_final.eng'
