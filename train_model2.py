import pandas as pd
import numpy as np



# load dataset
english = pd.read_csv('en.csv', names=['eng_sentence'])
french = pd.read_csv('fr.csv', names=['fr_sentence'])

df = pd.concat([english, french], axis=1)

df.columns = ['english', 'french']





import regex as re
def remove_pun (text):
	return re.sub(r'[.!?:;,]','', text)

df['english'] = df['english'].apply(lambda x: remove_pun(x))
df['french'] = df['french'].apply(lambda x: remove_pun(x))

df['FR Length']= df['french'].apply(lambda x: len(x.split()))
df['ENG Length']= df['english'].apply(lambda x: len(x.split()))

max_eng= max(df['ENG Length'])
max_fr= max(df['FR Length'])


# Tokenize the sentences that we have.
from tensorflow.keras.preprocessing.text import Tokenizer

eng_token = Tokenizer()
eng_token.fit_on_texts(df['english'])


fr_token = Tokenizer()
fr_token.fit_on_texts(df['french'])

fr_tokenized = fr_token.texts_to_sequences(df['french'])
eng_tokenized = eng_token.texts_to_sequences(df['english'])



eng_vocab= len(eng_token.word_index)+1
fr_vocab = len(fr_token.word_index)+1

fr_tokenized_lengths = [len(sentence) for sentence in fr_tokenized]
eng_tokenized_lengths=[len(sentence) for sentence in eng_tokenized]


from tensorflow.keras.preprocessing.sequence import pad_sequences
eng_padded = pad_sequences(eng_tokenized, maxlen=max_eng, padding='post')
fr_padded = pad_sequences(fr_tokenized, maxlen=max_fr, padding='post')

#build the model 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, RepeatVector, TimeDistributed, Bidirectional, Input
max_seq_len_fr = max(fr_tokenized_lengths)
max_seq_len_eng = max(eng_tokenized_lengths)
model2 = Sequential()
model2.add(Embedding(eng_vocab, 100, mask_zero=True, input_shape=(max_seq_len_eng,)))
model2.add(Bidirectional(LSTM(256, return_sequences=False), merge_mode='concat'))
model2.add(RepeatVector(max_seq_len_fr))
model2.add(Bidirectional(LSTM(256, return_sequences=True), merge_mode='concat'))
model2.add(TimeDistributed(Dense(fr_vocab, activation='softmax')))


model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model2.summary()


from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 10)
final_model2 = model2.fit(eng_padded, fr_padded, epochs=50, batch_size=512, validation_split = 0.3, callbacks=[early_stop])

# Save the tokenizers
eng_token_json = eng_token.to_json()
with open('eng_tokenizer.json', 'w') as f:
    f.write(eng_token_json)

fr_token_json = fr_token.to_json()
with open('fr_tokenizer.json', 'w') as f:
    f.write(fr_token_json)


def translate2(input, tokenizer):
    input = [input]
    test_tokenized = eng_token.texts_to_sequences(input)
    test_padded = pad_sequences(test_tokenized, maxlen=max_seq_len_eng, padding='post')

    predictions = model2.predict(test_padded)[0]

    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ''
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(predictions, 1)])

input = "she is driving the truck"

#Test Your Zaka


print('English: '+input)
print('French: elle conduit le camion')
print('Predicted: '+translate2(input, fr_token))
model2.save('model2.translator.h5')
