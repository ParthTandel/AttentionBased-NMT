import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from utils import Utils
import numpy as np


def EncodeDecoderModel(num_english_token, num_french_token, num_hidden_state, max_encoder_len, max_decoder_len ):

    encoder_inputs = Input(shape=(None, ))
    encoder_embedding = Embedding(input_dim=num_english_token, output_dim=200, mask_zero=True )(encoder_inputs)
    encoder = LSTM(num_hidden_state, return_state=True, return_sequences=True)
    _, state_h, state_c = encoder(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(input_dim=num_french_token, output_dim=200, mask_zero=True )(decoder_inputs)
    decoder_lstm = LSTM(num_hidden_state, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(num_french_token, activation='softmax')

    decoder_outputs = decoder_dense(decoder_outputs)

    print("decoder shape")
    print(decoder_outputs.shape)
    print("decoder shape")

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    # inference model
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.summary()

    # # enc_model = Model(encoder_inputs, enc_state)
    # # enc_model.summary()

    decoder_inputs_state_h = Input(shape=(num_hidden_state,))
    decoder_inputs_state_c = Input(shape=(num_hidden_state,))
    decoder_state_input = [decoder_inputs_state_h, decoder_inputs_state_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_state_input)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs, decoder_state_input], [decoder_outputs, decoder_states])
    decoder_model.summary()
    return model, encoder_model, decoder_model
    # return model


def predictSequence(input_seq, encoder_model, decoder_model, frn_vocab, frn_reverse_vocab, decoder_seq_len):
    num_french_token = len(frn_reverse_vocab.keys())
    states_value = encoder_model.predict(np.array([input_seq]))
    print(states_value[0].shape)
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = frn_vocab["initchar"]
    print(target_seq)
    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # print(output_tokens.shape)
        sampled_token_index = np.argmax(output_tokens[0, 0, :])


        sampled_char = " " + frn_reverse_vocab[sampled_token_index]
        # print(sampled_char, end=" ")
        decoded_sentence.append(sampled_char)


        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

        if len(decoded_sentence) > 11 or sampled_token_index == frn_vocab["endchar"]:
            stop_condition = True

    print(" ".join(decoded_sentence))

    return decoded_sentence


def main():
    batch_size = 200  # Batch size for training.
    epochs = 10  # Number of epochs to train for.
    hidden_state_dim = 256  # dimensionality of the hidden space.
    data_path = 'data/conv.txt'
    util = Utils()

    encoder_data, decoder_data, decoder_target, eng_vocab, eng_reverse_vocab, frn_vocab, frn_reverse_vocab = util.loadData(data_path)

    model, encoder_model, decoder_model = EncodeDecoderModel(len(eng_vocab.keys()) + 1, len(frn_vocab.keys()) + 1,
                                                             hidden_state_dim, encoder_data.shape[1], decoder_data.shape[1])

    model.fit([encoder_data, decoder_data], decoder_target, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    for xx in range(len(encoder_data[0:10])):
        for l in encoder_data[xx:xx + 1]:
            for arr in l:
                # i = np.where(arr==1)[0][0]
                if arr == 0:
                    continue
                print(arr, eng_reverse_vocab[arr])

            print()

        for l in decoder_data[xx:xx + 1]:
            for arr in l:
                if arr == 0:
                    continue

                # i = np.where(arr==1)[0][0]
                i = arr
                print(i, frn_reverse_vocab[i])
            print()

        print(predictSequence(encoder_data[xx], encoder_model, decoder_model, frn_vocab, frn_reverse_vocab, decoder_data.shape[1]))

if __name__ == "__main__":
    main()
    pass
