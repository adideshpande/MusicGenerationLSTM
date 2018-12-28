import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Embedding
#This function builds the architecture of the LSTM Model
#This is the same function as applied to training data in train_model.py
def build_gen_model(no_of_unique):
    model = Sequential()
    model.add(Embedding(input_dim=no_of_unique, output_dim=512, batch_input_shape=(1,1)))
    model.add(LSTM(256, return_sequences=True, stateful=True))
    model.add(Dropout(0.2))
    #model.add(LSTM(256, return_sequences=True, stateful=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(256, stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(no_of_unique,activation='softmax'))
    return model

#This model generates input characters for the model
def generating_seq(idx,len_sequence):
    with open("char_index_nd.json") as f:
        char_index = json.load(f)
    index_char = {index: char for (char, index) in char_index.items()}
    no_of_unique = len(index_char)
    model = build_gen_model(no_of_unique)
    model.load_weights("Models/nd_s_cce_adam_256_256_16_64_40.h5")

    sequence = [idx]

    for i in range(len_sequence):
        b = np.zeros((1,1))
        #b = np.array(sequence)
        #b[0,0] = sequence[-3]
        #b[0,1] = sequence[-2]
        b[0,0] = sequence[-1]
        pred_proba = model.predict_on_batch(b).ravel()
        out = np.random.choice(range(no_of_unique), size=1, p=pred_proba)
        sequence.append(out)

    seq_out = ''.join(index_char[i] for i in sequence)

    count = 0
    for char in seq_out:
        count+=1
        if char == '\n':
            break
    seq_out = seq_out[count:]
    '''
    count = 0
    for char in seq_out:
        count += 1
        if char == '\n' and seq_out[count] == '\n':
            break
    seq_out = seq_out[:count]
    '''
    return seq_out

music = generating_seq(30,5)
print(music)
