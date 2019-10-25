from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import random

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
            for word in text.lower().split():
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

    def loadData(self, data_path = "data/fra.txt"):

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        encoder_data = []    
        decoder_data = []
        for line in lines:
            try:
                line = line.split("\t")
                encoder,decoder = line[0], line[1]
                encoder_data.append(encoder)
                decoder_data.append(decoder)
            except: 
                pass

        encoder_sequence, encoder_vocab, encoder_reverse_vocab = self.tokenize(encoder_data)
        decoder_sequence, decoder_vocab, decoder_reverse_vocab = self.tokenize(decoder_data)
        decoder_sequence = [[2] + x + [3] for x in decoder_sequence]
        decoder_sequence_target = [ x[1:] for x in decoder_sequence]

        num_decoder_data_token = max([max(x) for x in decoder_sequence]) + 1
        num_encoder_data_token = max([max(x) for x in encoder_sequence]) + 1

        len_decoder_data_sen = max([len(x) for x in decoder_sequence]) + 2
        len_encoder_data_sen = max([len(x) for x in encoder_sequence]) + 2

        encoder_sequence = self.padSequence(encoder_sequence, len_encoder_data_sen)
        decoder_sequence = self.padSequence(decoder_sequence, len_decoder_data_sen)
        decoder_sequence_target = self.padSequence(decoder_sequence_target, len_decoder_data_sen)
        decoder_sequence_target = self.oneHotEncoder(decoder_sequence_target, num_decoder_data_token)

        return encoder_sequence, decoder_sequence, decoder_sequence_target, encoder_vocab, encoder_reverse_vocab, decoder_vocab, decoder_reverse_vocab

