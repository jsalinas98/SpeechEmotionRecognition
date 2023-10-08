from SpeechEmotionRecognizer import SpeechEmotionRecognizer
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

class SER_CNN(SpeechEmotionRecognizer):

    def __init__(self):
        super().__init__()

    def dataProcess(self, features):
        
        # extracting features    
        result = []               
        count = 0
        for audioData in self.audios: 
            extractedFeatures = np.array([])
            for feature in features:
                extractedFeatures = np.hstack((extractedFeatures, self.extractFeatures(feature, audioData)))
            result.append(extractedFeatures)
            print('audios feature extracted: {}/{}'.format(count, len(self.audios)), end="\r")
            count+=1
        print('\n')
        print('features extracted correctly!'.format(feature))

        self.X = np.array(result)
        # one hot encoding labels
        encoder = OneHotEncoder()
        self.Y = encoder.fit_transform(np.array(self.labels).reshape(-1,1)).toarray()

        # normalize data
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        self.X = np.expand_dims(self.X, axis=2)

        print(self.X.shape)

    def extractFeatures(self, feature, data):
        # ZCR
        if feature == 'zfr':
            result = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)

        # Chroma_stft
        elif feature == 'Chroma_stft':
            stft = np.abs(librosa.stft(data))
            result = np.mean(librosa.feature.chroma_stft(S=stft, sr=self.sampleRate).T, axis=0)

        # MFCC
        elif feature == 'mfcc':
            result = np.mean(librosa.feature.mfcc(y=data, sr=self.sampleRate).T, axis=0)

        # Root Mean Square Value
        elif feature == 'rms':
            result = np.mean(librosa.feature.rms(y=data).T, axis=0)

        # MelSpectogram
        elif feature == 'mel':
            result = np.mean(librosa.feature.melspectrogram(y=data, sr=self.sampleRate).T, axis=0)

        return result

    def createModel(self):
        self.model=Sequential()
        self.model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(self.X.shape[1], 1)))
        self.model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        self.model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        self.model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        self.model.add(Dropout(0.2))

        self.model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        self.model.add(Flatten())
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(units=8, activation='softmax'))
        self.model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])


    def train(self):
        # spliting data
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=0, shuffle=True, test_size=self.TrainValidationSplit)

        rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
        self.history=self.model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])

    def test(self):
        pass

    def predict(self):
        pass

recognizer = SER_CNN()

dataset = pd.read_csv('C:\\Users\\jsali\\OneDrive - UNIVERSIDAD DE SEVILLA\\Universidad\\MIERA\\TFM_SER\\dataset.csv')

recognizer.loadData(dataset.path, dataset.emotion)

recognizer.dataProcess(['mfcc', 'mel'])

recognizer.createModel()

recognizer.train()