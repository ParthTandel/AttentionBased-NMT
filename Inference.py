import numpy as np
from Model import EncodeDecoderModel
import json

class Inference:

    def __init__(self, meta_model_file,\
            decoder_vocab, decoder_reverse_vocab,\
            encoder_vocab, encoder_reverse_vocab,\
            encoder_model_file, decoder_model_file):

        self.decoder_vocab = decoder_vocab
        self.encoder_vocab = encoder_vocab
        self.decoder_reverse_vocab = decoder_reverse_vocab
        self.encoder_reverse_vocab = encoder_reverse_vocab

        with open(meta_model_file, "r") as fl:
            js = json.load(fl)


        num_encoder_token   = js["max_encoder_vocab"] + 1
        num_decoder_token   = js["max_decoder_vocab"] + 1
        max_encoder_len     = js["encoder_len"]
        max_decoder_len     = js["decoder_len"]
        hidden_state_dim = 200
        
        self.encoder_len = max_encoder_len

        _, encoder_model, decoder_model = EncodeDecoderModel( num_encoder_token ,
                                                                num_decoder_token,
                                                                hidden_state_dim,
                                                                max_encoder_len,
                                                                max_decoder_len)

        encoder_model.load_weights(encoder_model_file)
        decoder_model.load_weights(decoder_model_file)

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def predictSequence(self, input):

        input_seq = [0] * self.encoder_len
        x = 0
        for i in input.lower():
            if i in self.encoder_vocab:
                input_seq[x] = self.encoder_vocab[i]
            else:
                input_seq[x] = self.encoder_vocab["unk"]

        states_value = self.encoder_model.predict(np.array([input_seq]))
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = self.decoder_vocab["initchar"]
        stop_condition = False
        decoded_sentence = []

        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, 0, :])
            sampled_char = "" + self.decoder_reverse_vocab[str(sampled_token_index)]
            decoded_sentence.append(sampled_char)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

            if len(decoded_sentence) > 11 or sampled_token_index == self.decoder_vocab["endchar"]:
                stop_condition = True

        print("".join(decoded_sentence))

        return sampled_char



if  __name__ == "__main__":
    meta_model_file = "modelData/meta_data.json"

    decoder_vocab = {}
    with open("modelData/decoder_vocab.json") as fl:
        decoder_vocab = json.load(fl)
    
    decoder_reverse_vocab = {}
    with open("modelData/decoder_reverse_vocab.json") as fl:
        decoder_reverse_vocab = json.load(fl)

    encoder_vocab = {}
    with open("modelData/encoder_vocab.json") as fl:
        encoder_vocab = json.load(fl)

    encoder_reverse_vocab = {}
    with open("modelData/encoder_reverse_vocab.json") as fl:
        encoder_reverse_vocab = json.load(fl)

    encoder_model_file = "savedModel/encoder_model.h5"
    decoder_model_file = "savedModel/decoder_model.h5"
    Inf = Inference(meta_model_file,\
            decoder_vocab, decoder_reverse_vocab,\
            encoder_vocab, encoder_reverse_vocab,\
            encoder_model_file, decoder_model_file)

    Inf.predictSequence("I cried.")

    pass