from SpeechEmotionRecognizer import SpeechEmotionRecognizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

    def train(self, test_report=False):  

        # spliting data
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=0, shuffle=True, test_size=self.TrainValidationSplit)

        self.model.fit(x_train, y_train)

        y_pred =  self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)

        if test_report:
            encoder = OneHotEncoder();
            encoder.fit_transform(np.array(self.labels).reshape(-1,1)).toarray()
            self.printConfusionMatrix(y_test, y_pred, encoder.categories_)

            print(classification_report(y_test, y_pred))

    # Nuevo extractor de features
    def zcr(self, data,frame_length,hop_length):
        zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(zcr)
    def rmse(self, data,frame_length=2048,hop_length=512):
        rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(rmse)
    def mfcc(self, data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
        mfcc=librosa.feature.mfcc(y=data,sr=sr)
        return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

    def extract_features(self, data,sr=22050,frame_length=2048,hop_length=512):
        result=np.array([])
        
        result=np.hstack((result,
                        self.zcr(data,frame_length,hop_length),
                        self.rmse(data,frame_length,hop_length),
                        self.mfcc(data,sr,frame_length,hop_length)
                        ))
        
        return result

    def get_features(self):

        signal_padded = self.pad_signal(self.audios)

        features=list(map(self.extract_features, signal_padded))
        
        self.X = np.array(features)
        
        self.Y = self.labels

        # normalize data
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def pad_signal(self, audios):

        signal = []
        sample_rate = 16000     
        # Max pad lenght (3.0 sec)
        max_pad_len = 49100

        for audio in audios:
                
            # Padding or truncated signal 
            if len(audio) < max_pad_len:    
                X_padded = np.zeros(max_pad_len)
                X_padded[:len(audio)] = audio
                audio = X_padded
            elif len(audio) > max_pad_len:
                audio = np.asarray(audio[:max_pad_len])

            signal.append(audio)

        return np.array(signal)

    def predict(self):
        pass

recognizer = SER_SVM()

dataset = pd.read_csv('C:\\Users\\jsali\\OneDrive - UNIVERSIDAD DE SEVILLA\\Universidad\\MIERA\\TFM_SER\\dataset.csv')

dataset = dataset[dataset['database']=='ravdess']

recognizer.loadData(dataset.path, dataset.emotion)

#recognizer.dataProcess(['mfcc','mel','rms','zfr','chroma_stft'])

recognizer.get_features()

recognizer.createModel()

recognizer.train(test_report=True)