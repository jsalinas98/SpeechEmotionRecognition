from SpeechEmotionRecognizer import SpeechEmotionRecognizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SER_SVM(SpeechEmotionRecognizer):

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
        self.Y = self.labels

        # normalize data
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def extractFeatures(self, feature, data):

        # ZCR
        if feature == 'zfr':
            result = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)

        # Chroma_stft
        elif feature == 'chroma_stft':
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

        self.model = LinearSVC()

    def train(self):  

        # spliting data
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=0, shuffle=True, test_size=self.TrainValidationSplit)

        self.model.fit(x_train, y_train)

        y_pred =  self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)

    def test(self):
        pass

    def predict(self):
        pass

recognizer = SER_SVM()

dataset = pd.read_csv('C:\\Users\\jsali\\OneDrive - UNIVERSIDAD DE SEVILLA\\Universidad\\MIERA\\TFM_SER\\dataset.csv')

recognizer.loadData(dataset.path, dataset.emotion)

recognizer.dataProcess(['mfcc','mel','rms','zfr','chroma_stft'])

recognizer.createModel()

recognizer.train()