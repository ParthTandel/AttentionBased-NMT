from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


class Utils:

    def tokenize(self, x):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x)
        sequences = tokenizer.texts_to_sequences(x)
        return sequences, tokenizer
    def padSequence(self, sequences, maxLen):
        returnSeq = []
        for seq in sequences:
            lenLeft = maxLen - len(seq)
            seq = seq + ([0] * lenLeft)
            returnSeq.append(seq)
        return returnSeq

    def oneHotEncoder(self, sequences, num_token):
        returnSeq = []
        for seq in sequences:
            vect = []
            for i in seq:
                x = [0] * num_token
                x[i] = 1
                vect.append(x)
            returnSeq.append(vect)
        return returnSeq
            

    def loadData(self, data_path = "data/fra.txt"):

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        english = []
        french = []
        for line in lines[0:1000]:
            line = line.split("\t")
            eng,frn = line[0], line[1]
            english.append(eng)
            french.append(frn)


        eng_sequence, eng_tokenizer = self.tokenize(english)
        frn_sequence, frn_tokenizer = self.tokenize(french)

        num_french_token = max([max(x) for x in frn_sequence]) + 1
        num_english_token = max([max(x) for x in eng_sequence]) + 1

        len_french_sen = max([len(x) for x in frn_sequence])
        len_english_sen = max([len(x) for x in eng_sequence])

        eng_sequence = self.padSequence(eng_sequence, len_english_sen)
        frn_sequence = self.padSequence(frn_sequence, len_french_sen)

        print( "num_french_token", "num_english_token", "len_french_sen", "len_english_sen")
        print( num_french_token, num_english_token, len_french_sen, len_english_sen)

        oneHotEngSeq = self.oneHotEncoder(eng_sequence, num_english_token)
        oneHotFrnSeq = self.oneHotEncoder(frn_sequence, num_french_token)

        oneHotFrnSeq_target = []

        for seq in oneHotFrnSeq:
            seq[0] =  [0] * len(seq[0])
            oneHotFrnSeq_target.append(seq)

        oneHotEngSeq = np.array(oneHotEngSeq)
        oneHotFrnSeq = np.array(oneHotFrnSeq)
        oneHotFrnSeq_target = np.array(oneHotFrnSeq_target)


        print(oneHotEngSeq.shape, oneHotFrnSeq.shape)

        return oneHotEngSeq, oneHotFrnSeq, oneHotFrnSeq_target, eng_tokenizer, frn_tokenizer

