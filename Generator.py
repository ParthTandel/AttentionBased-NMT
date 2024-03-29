import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        encoder_data = []
        for index in list_IDs_temp:
            arr = np.load('modelData/encoder_npz/encoder_input_' + str(index) + '.npy')
            encoder_data.append(arr)
        encoder_data = np.array(encoder_data)

        decoder_data = []
        for index in list_IDs_temp:
            arr = np.load('modelData/decoder_npz/decoder_input_' + str(index) + '.npy')
            decoder_data.append(arr)
        decoder_data = np.array(decoder_data)

        decoder_target_data = []
        for index in list_IDs_temp:
            arr = np.load('modelData/decoder_target_npz/decoder_target_' + str(index) + '.npy')
            decoder_target_data.append(arr)
        decoder_target_data = np.array(decoder_target_data)

        return [encoder_data, decoder_data], decoder_target_data