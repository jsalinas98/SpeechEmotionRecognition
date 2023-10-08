from SpeechEmotionRecognizer import SpeechEmotionRecognizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import pandas as pd
import librosa
from math import ceil
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras import backend as K 
from keras.layers import Input, TimeDistributed, LSTM
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

class SER_CNN_LSTM(SpeechEmotionRecognizer):

    def dataProcess(self):
        self.audios = self.pad_signal()
        mel_spec = np.asarray(list(map(self.mel_spectrogram, self.audios)))
        self.X = np.asarray(list(map(self.frame, mel_spec)))

        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1] , self.X.shape[2], self.X.shape[3], 1)

        encoder = OneHotEncoder()
        self.Y = encoder.fit_transform(np.array(self.labels).reshape(-1,1)).toarray()

        print('Data processed correctly!')

    def pad_signal(self):

        signal = []
        sample_rate = 16000     
        # Max pad lenght (3.0 sec)
        max_pad_len = 49100

        for audio in self.audios:
                
            # Padding or truncated signal 
            if len(audio) < max_pad_len:    
                X_padded = np.zeros(max_pad_len)
                X_padded[:len(audio)] = audio
                audio = X_padded
            elif len(audio) > max_pad_len:
                audio = np.asarray(audio[:max_pad_len])

            signal.append(audio)

        return signal

    # Compute mel spectogram
    def mel_spectrogram(self, data, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000, graph=False):
                
        mel_spect = np.abs(librosa.stft(data, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2    
        mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax) 
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        
        if (graph):
            fig, ax = plt.subplots(nrows=1, sharex=True)

            img = librosa.display.specshow(mel_spect,
                                        x_axis='time', y_axis='mel', fmax=8000)
            fig.colorbar(img)
            ax.set(title='Mel spectrogram')
            ax.label_outer()
        
        return mel_spect

    # Dividimos datos en frames
    def frame(self, x, win_step=64, win_size=128):
        n_frames = ceil(x.shape[1] / win_step)
        frames = np.zeros((n_frames, x.shape[0], win_size)).astype(np.float32)
        x_padded = np.zeros((x.shape[0], win_step*(n_frames-1)+win_size))
        x_padded[:,:x.shape[1]] = x
        for t in range(n_frames):
            frames[t,:,:] = np.copy(x_padded[:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)

        return frames

    def createModel(self):
        # Define two sets of inputs: MFCC and FBANK
        input_y = Input(shape=self.X.shape[1:], name='Input_MELSPECT')

        ## First LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT')(input_y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)     

        ## Second LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)

        ## Second LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

        ## Second LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)  

        ## Flat
        y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)                      
                                    
        # Apply 2 LSTM layer and one FC
        y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)
        y = Dense(self.Y.hape[1], activation='softmax', name='FC')(y)

        # Build final model
        self.model = Model(inputs=input_y, outputs=y)

    def train(self):
        # spliting data
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=0, shuffle=True, test_size=self.TrainValidationSplit)

        self.model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-6, momentum=0.8), loss='categorical_crossentropy', metrics=['accuracy'])
        # Fit model
        history = self.model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))

        self.model.save('/Models/CNN_LSTM_001.h5')

    def test(self):
        pass

    def predict(self):
        pass

recognizer = SER_CNN_LSTM()

dataset = pd.read_csv('C:\\Users\\jsali\\OneDrive - UNIVERSIDAD DE SEVILLA\\Universidad\\MIERA\\TFM_SER\\dataset.csv')

recognizer.loadData(dataset.path, dataset.emotion)

recognizer.dataProcess()

recognizer.createModel()

recognizer.train()º
