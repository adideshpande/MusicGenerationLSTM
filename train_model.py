import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
import io

b_s = 16
len_sequence = 64
epochs = 80
def read_batches(data, no_of_unique):
    batch_char = int(data.shape[0]/b_s)
    for i in range(0,batch_char - len_sequence, len_sequence):
        x = np.zeros((b_s,len_sequence))
        y = np.zeros((b_s,len_sequence,no_of_unique))
        for row in range(b_s):
            for col in range(len_sequence):
                x[row, col] = data[row*batch_char + i + row]
                y[row, col, data[row * batch_char + i + row + 1]] = 1
        yield x,y

def build_the_model(batch_size, sequence_length, unique_chars):
    model = Sequential()
    model.add(Embedding(input_dim = unique_chars, output_dim = 512, batch_input_shape = (batch_size, sequence_length)))
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(unique_chars,activation='softmax')))
    #model.add(Activation("softmax"))
    return model

def train_model(data, epoch):
    char_index = {char: index for (index, char) in enumerate(sorted(list(set(data))))}
    with open("char_index_nd.json",'w') as file:
        json.dump(char_index,file)
    no_of_unique = len(char_index)
    m = build_the_model(b_s,len_sequence, no_of_unique)
    #m.summary()
    m.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    chars = np.asarray([char_index[i] for i in data],dtype=np.int32)
    for e in range(epoch):
        print("Epoch "+str(e+1)+"/"+str(epoch))
        for i,(x,y) in enumerate(read_batches(chars,no_of_unique)):
            loss = list()
            acc = list()
            epoch_loss, epoch_accuracy = m.train_on_batch(x,y)
            loss.append(epoch_loss)
            acc.append(epoch_accuracy)
            print("Batch:"+str(i+1)+", Loss:"+str(epoch_loss)+", Accuracy:"+str(epoch_accuracy))
        final_loss.append(min(loss))
        final_acc.append(max(acc))
        if e+1 == epoch:
            m.save_weights("Models/nd_s_cce_adam_256_256_16_64_40.h5")
            print("Weights Saved")

final_loss = list()
final_acc = list()
with io.open("Data_Tunes.txt", 'r', encoding='utf8') as f:
    data = f.read()
train_model(data,epochs)
for i in range(len(final_loss)):
    print("Epoch Number:"+str(i+1)+", Loss:"+str(final_loss[i]+", Accuracy:"+str(final_acc[i])))