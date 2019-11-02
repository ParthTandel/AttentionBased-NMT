import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import json
from pprint import pprint
class Utils:

    def tokenize(self, x):
        index = 4
        vocab = {}
        reverse_vocab = {}
        vocab["unk"] = 1
        vocab["initchar"] = 2
        vocab["endchar"] = 3
        reverse_vocab[1] = "unk"
        reverse_vocab[2] = "initchar"
        reverse_vocab[3] = "endchar"

        sequences = []
        for text in x:
            seq = []
            for word in text.lower():
                if word not in vocab:
                    vocab[word] = index
                    reverse_vocab[index] = word
                    index = index + 1
                seq.append(vocab[word])
            sequences.append(seq)
        return np.array(sequences), vocab, reverse_vocab

    def padSequence(self, sequences, maxLen):
        returnSeq = []
        for seq in sequences:
            lenLeft = maxLen - len(seq)
            seq = seq + ([0] * lenLeft)
            returnSeq.append(seq)
        return np.array(returnSeq)

    def oneHotEncoder(self, sequences, num_token, skip=0):
        returnSeq = np.zeros((len(sequences), len(sequences[0]), num_token), dtype='float32')
        k = 0
        for seq in sequences:
            j = 0
            for i in seq:
                returnSeq[k][j][i] = 1
                j = j + 1
            k = k + 1
        return np.array(returnSeq)

    def loadData(self, data_path = "data/conv.txt"):

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        encoder_data = []    
        decoder_data = []
        for line in lines[0:min(20000, len(lines))]:
            try:
                line = line.split("\t")
                encoder,decoder = line[0], line[1]
                encoder_data.append(encoder)
                decoder_data.append(decoder)
            except: 
                pass

        encoder_sequence, encoder_vocab, encoder_reverse_vocab = self.tokenize(encoder_data)
        len_encoder_data_sen = max([len(x) for x in encoder_sequence]) + 2
        encoder_sequence = self.padSequence(encoder_sequence, len_encoder_data_sen)
        encoder_sequence_len = len(encoder_sequence)
        encoder_vocab_len = len(encoder_vocab.keys())

        index = 0
        for val in encoder_sequence:
            np.save('modelData/encoder_npz/encoder_input_' +str(index) , val)
            index = index + 1
        del encoder_sequence

        with open("modelData/encoder_vocab.json", "w") as fl:
            json.dump(encoder_vocab, fl)
        del encoder_vocab

        with open("modelData/encoder_reverse_vocab.json", "w") as fl:
            json.dump(encoder_reverse_vocab, fl)
        del encoder_reverse_vocab

        decoder_sequence, decoder_vocab, decoder_reverse_vocab = self.tokenize(decoder_data)
        decoder_sequence = [[2] + x + [3] for x in decoder_sequence]
        decoder_sequence_target = [ x[1:] for x in decoder_sequence]

        num_decoder_data_token = max([max(x) for x in decoder_sequence]) + 1
        len_decoder_data_sen = max([len(x) for x in decoder_sequence]) + 2
        decoder_vocab_len = len(decoder_vocab.keys())

        decoder_sequence = self.padSequence(decoder_sequence, len_decoder_data_sen)

        index = 0
        for val in decoder_sequence:
            np.save('modelData/decoder_npz/decoder_input_' +str(index) , val)
            index = index + 1
        del decoder_sequence

        decoder_sequence_target = self.padSequence(decoder_sequence_target, len_decoder_data_sen)
        decoder_sequence_target = self.oneHotEncoder(decoder_sequence_target, num_decoder_data_token)

        index = 0
        for val in decoder_sequence_target:
            np.save('modelData/decoder_target_npz/decoder_target_' +str(index) , val)
            index = index + 1

        del decoder_sequence_target

        with open("modelData/decoder_vocab.json", "w") as fl:
            json.dump(decoder_vocab, fl)
        del decoder_vocab

        with open("modelData/decoder_reverse_vocab.json", "w") as fl:
            json.dump(decoder_reverse_vocab, fl)
        del decoder_reverse_vocab


        train_ids = list(np.random.choice(encoder_sequence_len, int(encoder_sequence_len * 0.8)))
        val_test_ids = list(set(range(encoder_sequence_len)) - set(train_ids))
        validation_ids = val_test_ids[0:int(len(val_test_ids)/2)]
        test_ids = val_test_ids[int(len(val_test_ids)/2):]

        meta_json = {
            "train_ids"         : [int(x) for x in train_ids],
            "validation_ids"    : [int(x) for x in validation_ids],
            "test_ids"          : [int(x) for x in test_ids],
            "len"               : encoder_sequence_len,
            "encoder_len"       : len_encoder_data_sen,
            "decoder_len"       : len_decoder_data_sen,
            "max_encoder_vocab" : encoder_vocab_len,
            "max_decoder_vocab" : decoder_vocab_len,
        }

        with open("modelData/meta_data.json", "w") as fl:
            json.dump(meta_json, fl)


        return True


