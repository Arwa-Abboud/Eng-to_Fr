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

#build the model 1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, RepeatVector, TimeDistributed
max_seq_len_fr = max(fr_tokenized_lengths)
max_seq_len_eng = max(eng_tokenized_lengths)
model = Sequential()
#encoder
model.add(Embedding(input_dim=eng_vocab, output_dim=128, input_shape=(max_seq_len_eng,)))
model.add(LSTM(256,  return_sequences=False)) #I added return_sequences=False in the encoder LSTM (meaning the model will output only the last hidden state)
# RepeatVector to replicate the encoder output for each time step of the decoder
model.add(RepeatVector(max_seq_len_fr))
# Decoder
model.add(LSTM(256,  return_sequences=True)) # true which allows it to generate sequences
model.add(TimeDistributed(Dense(fr_vocab, activation='softmax')))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

from keras.callbacks import  EarlyStopping

early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 10)
final_model = model.fit(eng_padded, fr_padded, epochs=20, batch_size=512, validation_split = 0.2, callbacks=[early_stop])

def translate(input, tokenizer):
    input = [input]
    test_tokenized = eng_token.texts_to_sequences(input)
    test_padded = pad_sequences(test_tokenized, maxlen=max_seq_len_eng, padding='post')

    predictions = model.predict(test_padded)[0]

    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ''
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(predictions, 1)])


input = "she is driving the truck"

#Test Your Zaka


print('English: '+input)
print('French: elle conduit le camion')
print('Predicted: '+translate(input,  fr_token))
model.save('model1_translator.h5')
